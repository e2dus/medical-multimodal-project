"""
Lead-field constrained 1-lead to selected-lead ECG reconstruction.

This is the stronger physics-oriented successor to the geometry-guided residual
model. The neural network predicts source activations J(t), then a fixed
or weakly learnable lead-field matrix L projects sources to ECG leads:

    V(t) = (L + dL_patient + dL_global) J(t)

If an ECGSIM/openCARP lead-field matrix is available, pass it as leadfield_init.
Otherwise a geometry-derived fallback matrix is used only as an initialization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLINICAL10_LEADS = ["I", "II", "III", "aVR", "aVL", "V1", "V2", "V4", "V5", "V6"]


def make_aha17_anchors() -> torch.Tensor:
    pts = torch.tensor(
        [
            [-0.030,  0.055,  0.030], [-0.048,  0.028,  0.032], [-0.048, -0.028,  0.032],
            [-0.030, -0.055,  0.030], [ 0.000, -0.060,  0.028], [ 0.000,  0.060,  0.028],
            [-0.015,  0.050,  0.000], [-0.035,  0.020,  0.002], [-0.035, -0.020,  0.002],
            [-0.015, -0.050,  0.000], [ 0.015, -0.050,  0.000], [ 0.015,  0.050,  0.000],
            [-0.005,  0.032, -0.030], [-0.022,  0.010, -0.032], [-0.022, -0.010, -0.032],
            [-0.005, -0.032, -0.030], [ 0.000,  0.000, -0.060],
        ],
        dtype=torch.float32,
    )
    return pts + torch.tensor([-0.03, 0.02, 0.10], dtype=torch.float32)


def make_electrode_positions() -> torch.Tensor:
    return torch.tensor(
        [
            [ 0.30,  0.25,  0.00], [-0.30,  0.25,  0.00], [-0.05, -0.40,  0.00],
            [ 0.03,  0.05,  0.20], [ 0.00,  0.05,  0.22], [-0.04,  0.01,  0.21],
            [-0.08, -0.02,  0.18], [-0.14, -0.02,  0.13], [-0.20, -0.02,  0.07],
        ],
        dtype=torch.float32,
    )


def parse_target_leads(spec: str) -> List[str]:
    if spec == "all":
        return LEADS
    if spec == "clinical10":
        return CLINICAL10_LEADS
    leads = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in leads if x not in LEADS]
    if unknown:
        raise ValueError(f"Unknown leads: {unknown}")
    return leads


def geometry_fallback_leadfield() -> torch.Tensor:
    """Approximate [12,17] lead-field from fixed source/electrode geometry."""
    src = make_aha17_anchors()
    elec = make_electrode_positions()
    r_vec = elec.unsqueeze(0) - src.unsqueeze(1)
    denom = (r_vec.pow(2).sum(dim=-1) + 0.04).pow(1.5)
    electrode_resp = (r_vec / denom.unsqueeze(-1)).abs().mean(dim=-1)

    v = electrode_resp
    ra, la, ll = v[:, 0], v[:, 1], v[:, 2]
    wct = (ra + la + ll) / 3.0
    limb = torch.stack(
        [
            la - ra,
            ll - ra,
            ll - la,
            ra - 0.5 * (la + ll),
            la - 0.5 * (ra + ll),
            ll - 0.5 * (ra + la),
        ],
        dim=0,
    )
    chest = (v[:, 3:9] - wct.unsqueeze(1)).T
    L = torch.cat([limb, chest], dim=0)
    return L / (L.std(dim=1, keepdim=True) + 1e-6)


def make_aha17_adjacency() -> torch.Tensor:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6),
        (12, 13), (13, 14), (14, 15), (15, 12),
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
        (6, 12), (7, 13), (8, 14), (9, 15), (16, 12), (16, 13), (16, 14), (16, 15),
    ]
    A = torch.zeros(17, 17)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    return A


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class LeadFieldLayer(nn.Module):
    def __init__(
        self,
        output_indices: List[int],
        leadfield_init: Optional[torch.Tensor] = None,
        learnable_delta: bool = True,
        delta_scale: float = 0.05,
    ):
        super().__init__()
        if leadfield_init is None:
            leadfield_init = geometry_fallback_leadfield()
        assert leadfield_init.ndim == 2 and leadfield_init.shape[0] == 12, "leadfield_init must be [12,N]"
        self.n_sources = int(leadfield_init.shape[1])
        self.register_buffer("L_base_full", leadfield_init.float())
        self.register_buffer("output_indices", torch.tensor(output_indices, dtype=torch.long))
        self.delta_scale = delta_scale
        if learnable_delta:
            self.L_delta = nn.Parameter(torch.zeros(12, self.n_sources))
        else:
            self.register_buffer("L_delta", torch.zeros(12, self.n_sources))

    def forward(self, source: torch.Tensor, patient_mod: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # source: [B,N,T], patient_mod: optional [B,L,N]
        L_full = self.L_base_full + torch.tanh(self.L_delta) * self.delta_scale
        L = L_full.index_select(0, self.output_indices.to(source.device))
        if patient_mod is not None:
            L = L.unsqueeze(0) + patient_mod
            ecg = torch.einsum("bls,bst->blt", L, source)
            return ecg, L
        ecg = torch.einsum("ls,bst->blt", L, source)
        return ecg, L.unsqueeze(0).expand(source.size(0), -1, -1)


class LeadFieldECGRecon(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 48,
        output_indices: Optional[List[int]] = None,
        leadfield_init: Optional[torch.Tensor] = None,
        learnable_delta: bool = True,
    ):
        super().__init__()
        if output_indices is None:
            output_indices = list(range(12))
        self.output_leads = [LEADS[i] for i in output_indices]
        self.out_ch = len(output_indices)
        if leadfield_init is None:
            n_sources = 17
        else:
            assert leadfield_init.ndim == 2 and leadfield_init.shape[0] == 12, "leadfield_init must be [12,N]"
            n_sources = int(leadfield_init.shape[1])
        self.n_sources = n_sources

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        # ── 기존 컴포넌트 전부 유지 (체크포인트 호환) ──
        self.source_head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(base_ch, self.n_sources, 1),
        )
        self.unet_residual_head = nn.Conv1d(base_ch, self.out_ch, 1)
        self.patient_mod_head = nn.Sequential(
            nn.Linear(base_ch * 8, base_ch * 2),
            nn.GELU(),
            nn.Linear(base_ch * 2, self.out_ch * self.n_sources),
        )
        self.gate_head = nn.Sequential(nn.Conv1d(base_ch, base_ch, 5, padding=2), nn.GELU(), nn.Conv1d(base_ch, self.out_ch, 1))
        self.leadfield = LeadFieldLayer(output_indices, leadfield_init=leadfield_init, learnable_delta=learnable_delta)
        self.register_buffer("aha17_anchors", make_aha17_anchors())
        self.register_buffer("aha17_adjacency", make_aha17_adjacency())
        self.physics_scale = nn.Parameter(torch.tensor(0.0))
        self.residual_scale = nn.Parameter(torch.tensor(-3.5))
        self.patient_mod_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.constant_(self.gate_head[-1].bias, -0.5)

    def forward(self, x: torch.Tensor, return_explain: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        source = self.source_head(d1)
        source = source - source.mean(dim=-1, keepdim=True)
        source = source / (source.std(dim=-1, keepdim=True) + 1e-6)

        pooled = b.mean(dim=-1)
        patient_mod = self.patient_mod_head(pooled).view(x.size(0), self.out_ch, self.n_sources)
        patient_mod = torch.tanh(patient_mod) * torch.sigmoid(self.patient_mod_scale)

        physics_ecg, L_patient = self.leadfield(source, patient_mod=patient_mod)
        physics_ecg = physics_ecg / (physics_ecg.std(dim=-1, keepdim=True) + 1e-6)
        gate = torch.sigmoid(self.gate_head(d1))
        unet_residual = torch.tanh(self.unet_residual_head(d1)) * torch.sigmoid(self.residual_scale)
        pred = torch.sigmoid(self.physics_scale) * gate * physics_ecg + unet_residual

        if not return_explain:
            return pred
        explain = {
            "source_activation": source,
            "leadfield_matrix": L_patient,
            "leadfield_base": self.leadfield.L_base_full,
            "physics_ecg": physics_ecg,
            "physics_gate": gate,
            "unet_residual": unet_residual,
            "aha17_anchors": self.aha17_anchors,
            "aha17_adjacency": self.aha17_adjacency,
            "output_leads": self.output_leads,
        }
        return pred, explain


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target) + 0.1 * F.mse_loss(pred, target)


def lead_consistency_loss(pred: torch.Tensor, lead_names: List[str]) -> torch.Tensor:
    idx = {lead: i for i, lead in enumerate(lead_names)}
    terms = []
    if all(k in idx for k in ("I", "II", "III")):
        terms.append((pred[:, idx["III"]] - (pred[:, idx["II"]] - pred[:, idx["I"]])).abs().mean())
    if all(k in idx for k in ("I", "II", "aVR")):
        terms.append((pred[:, idx["aVR"]] + 0.5 * (pred[:, idx["I"]] + pred[:, idx["II"]])).abs().mean())
    if all(k in idx for k in ("I", "II", "aVL")):
        terms.append((pred[:, idx["aVL"]] - (pred[:, idx["I"]] - 0.5 * pred[:, idx["II"]])).abs().mean())
    if all(k in idx for k in ("I", "II", "aVF")):
        terms.append((pred[:, idx["aVF"]] - (pred[:, idx["II"]] - 0.5 * pred[:, idx["I"]])).abs().mean())
    return torch.stack(terms).mean() if terms else pred.new_tensor(0.0)


def source_regularization_loss(explain: Dict[str, torch.Tensor], input_ecg: torch.Tensor) -> torch.Tensor:
    source = explain["source_activation"]
    temporal = torch.diff(source, dim=-1).abs().mean() + 0.5 * torch.diff(source, n=2, dim=-1).abs().mean()
    if source.size(1) == 17:
        A = explain["aha17_adjacency"].to(source.device)
        src_i = source.unsqueeze(2)
        src_j = source.unsqueeze(1)
        spatial = ((src_i - src_j).pow(2) * A.view(1, 17, 17, 1)).sum() / (A.sum() * source.size(0) * source.size(-1) + 1e-6)
    else:
        spatial = source.new_tensor(0.0)

    diff = torch.diff(input_ecg, dim=-1, prepend=input_ecg[..., :1])
    qrs = F.avg_pool1d(diff.pow(2), kernel_size=31, stride=1, padding=15)
    qrs = qrs / (qrs.amax(dim=-1, keepdim=True) + 1e-6)
    source_energy = source.abs().mean(dim=1, keepdim=True)
    source_energy = source_energy / (source_energy.amax(dim=-1, keepdim=True) + 1e-6)
    qrs_align = F.l1_loss(source_energy, qrs)
    return 0.2 * temporal + 0.1 * spatial + 0.5 * qrs_align


def leadfield_regularization_loss(explain: Dict[str, torch.Tensor]) -> torch.Tensor:
    L = explain["leadfield_matrix"]
    base = explain["leadfield_base"].to(L.device)
    output_leads = explain.get("output_leads", LEADS)
    row_idx = torch.tensor([LEADS.index(x) for x in output_leads], device=L.device)
    base = base.index_select(0, row_idx)
    return F.mse_loss(L.mean(dim=0), base)
