"""
Explainable 1-lead to 12-lead ECG reconstruction.

This model keeps the reconstruction path simple: a 1D U-Net produces shared
temporal features, then each output lead is reconstructed as a learned mixture
of a small set of latent waveform bases. The mixture weights and temporal
importance maps are returned as explanations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


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


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.block = ConvBlock(out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.skip_gate = nn.Sequential(nn.Conv1d(skip_ch, skip_ch, kernel_size=1), nn.Sigmoid())
        self.block = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        gate = self.skip_gate(skip)
        return self.block(torch.cat([x, skip * gate], dim=1)), gate


class ExplainableECGRecon(nn.Module):
    """
    U-Net reconstruction model with a faithful latent-basis explanation head.

    Forward returns:
      pred: [B, 12, T]
      explain:
        basis: [B, K, T], shared latent waveform bases
        lead_mix: [B, 12, K], signed contribution weights per lead
        basis_contribution: [B, 12, K, T], per-lead temporal decomposition
        temporal_importance: [B, 1, T], where the decoder focused
        skip_gates: low/mid/high U-Net skip usage maps
        lead_rule_residuals: Einthoven/Goldberger consistency residuals
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 12,
        base_ch: int = 48,
        n_basis: int = 8,
        output_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        if output_indices is None:
            output_indices = list(range(out_ch))
        self.output_leads = [LEADS[i] for i in output_indices]
        self.out_ch = len(output_indices)
        self.n_basis = n_basis

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.basis_head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(base_ch, n_basis, kernel_size=1),
        )
        self.residual_head = nn.Conv1d(base_ch, self.out_ch, kernel_size=1)
        self.temporal_head = nn.Sequential(nn.Conv1d(base_ch, 1, kernel_size=1), nn.Sigmoid())
        self.source_activation_head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(base_ch, 17, kernel_size=1),
            nn.Softplus(),
        )
        self.source_vector_head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(base_ch, 17 * 3, kernel_size=1),
        )
        self.source_to_lead_head = nn.Sequential(
            nn.Linear(base_ch * 8, base_ch * 2),
            nn.GELU(),
            nn.Linear(base_ch * 2, self.out_ch * 17),
        )
        self.physics_gate_head = nn.Sequential(
            nn.Conv1d(base_ch, base_ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(base_ch, self.out_ch, kernel_size=1),
        )
        self.register_buffer("aha17_anchors", make_aha17_anchors())
        self.register_buffer("electrode_positions", make_electrode_positions())
        self.register_buffer("output_indices_tensor", torch.tensor(output_indices, dtype=torch.long))

        self.mix_head = nn.Sequential(
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.GELU(),
            nn.Linear(base_ch * 4, self.out_ch * n_basis),
        )
        self.mix_scale = nn.Parameter(torch.tensor(1.0))
        self.residual_scale = nn.Parameter(torch.tensor(-2.0))
        self.physics_scale = nn.Parameter(torch.tensor(-1.0))
        nn.init.constant_(self.physics_gate_head[-1].bias, -1.0)

    def _physics_render(self, source_activation: torch.Tensor, source_vector: torch.Tensor) -> torch.Tensor:
        B, _, T = source_activation.shape
        src = self.aha17_anchors.to(source_activation.device)
        elec = self.electrode_positions.to(source_activation.device)
        r_vec = elec.unsqueeze(0) - src.unsqueeze(1)
        denom = ((r_vec.pow(2).sum(dim=-1) + 0.04) ** 1.5).unsqueeze(-1)
        moment = source_vector * source_activation.unsqueeze(2)
        v_per_source = (moment.unsqueeze(2) * r_vec.unsqueeze(0).unsqueeze(-1)).sum(dim=3) / denom.unsqueeze(0)
        v_elec = v_per_source.sum(dim=1).clamp(-100.0, 100.0)
        v_ra, v_la, v_ll = v_elec[:, 0], v_elec[:, 1], v_elec[:, 2]
        wct = (v_ra + v_la + v_ll) / 3.0
        limb = torch.stack(
            [
                v_la - v_ra,
                v_ll - v_ra,
                v_ll - v_la,
                v_ra - 0.5 * (v_la + v_ll),
                v_la - 0.5 * (v_ra + v_ll),
                v_ll - 0.5 * (v_ra + v_la),
            ],
            dim=1,
        )
        chest = v_elec[:, 3:9] - wct.unsqueeze(1)
        phys = torch.cat([limb, chest], dim=1)
        return phys / (phys.std(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor, return_explain: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        d3, gate3 = self.up3(b, e3)
        d2, gate2 = self.up2(d3, e2)
        d1, gate1 = self.up1(d2, e1)

        basis = self.basis_head(d1)
        pooled = b.mean(dim=-1)
        lead_mix = self.mix_head(pooled).view(x.size(0), self.out_ch, self.n_basis)
        lead_mix = torch.tanh(lead_mix) * self.mix_scale.abs()

        basis_recon = torch.einsum("blk,bkt->blt", lead_mix, basis)
        residual = torch.tanh(self.residual_head(d1)) * torch.sigmoid(self.residual_scale)
        unet_pred = basis_recon + residual

        source_activation = self.source_activation_head(d1)
        source_activation = source_activation / (source_activation.mean(dim=1, keepdim=True) + 1e-6)
        source_vector_raw = torch.tanh(self.source_vector_head(d1)).view(x.size(0), 17, 3, x.size(-1))
        source_vector = F.normalize(source_vector_raw, dim=2, eps=1e-6)
        source_to_lead = self.source_to_lead_head(pooled).view(x.size(0), self.out_ch, 17)
        physics_raw_full = self._physics_render(source_activation, source_vector)
        physics_raw = physics_raw_full.index_select(1, self.output_indices_tensor.to(physics_raw_full.device))
        physics_gate = torch.sigmoid(self.physics_gate_head(d1))
        physics_residual = physics_gate * torch.sigmoid(self.physics_scale) * physics_raw
        pred = unet_pred + physics_residual

        if not return_explain:
            return pred

        explain = {
            "basis": basis,
            "lead_mix": lead_mix,
            "basis_contribution": lead_mix.unsqueeze(-1) * basis.unsqueeze(1),
            "temporal_importance": self.temporal_head(d1),
            "aha17_anchors": self.aha17_anchors,
            "electrode_positions": self.electrode_positions,
            "source_activation": source_activation,
            "source_vector": source_vector,
            "source_to_lead": torch.tanh(source_to_lead),
            "unet_pred": unet_pred,
            "output_leads": self.output_leads,
            "physics_raw_full": physics_raw_full,
            "physics_raw": physics_raw,
            "physics_residual": physics_residual,
            "physics_gate": physics_gate,
            "skip_gates": {"high": gate3, "mid": gate2, "low": gate1},
            "lead_rule_residuals": lead_rule_residuals(pred, self.output_leads),
            "residual_fraction": residual.abs().mean(dim=(1, 2)) / (pred.abs().mean(dim=(1, 2)) + 1e-6),
        }
        return pred, explain


def lead_rule_residuals(ecg: torch.Tensor, lead_names: List[str] = LEADS) -> Dict[str, torch.Tensor]:
    """Return available standard-lead consistency residuals for [B, L, T]."""
    idx = {lead: i for i, lead in enumerate(lead_names)}
    out = {}
    if all(k in idx for k in ("I", "II", "III")):
        out["III_minus_II_minus_I"] = ecg[:, idx["III"]] - (ecg[:, idx["II"]] - ecg[:, idx["I"]])
    if all(k in idx for k in ("I", "II", "aVR")):
        out["aVR_plus_half_I_II"] = ecg[:, idx["aVR"]] + 0.5 * (ecg[:, idx["I"]] + ecg[:, idx["II"]])
    if all(k in idx for k in ("I", "II", "aVL")):
        out["aVL_minus_I_plus_half_II"] = ecg[:, idx["aVL"]] - (ecg[:, idx["I"]] - 0.5 * ecg[:, idx["II"]])
    if all(k in idx for k in ("I", "II", "aVF")):
        out["aVF_minus_II_plus_half_I"] = ecg[:, idx["aVF"]] - (ecg[:, idx["II"]] - 0.5 * ecg[:, idx["I"]])
    return out


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, input_idx: int) -> torch.Tensor:
    mask = torch.ones(pred.size(1), dtype=torch.bool, device=pred.device)
    mask[input_idx] = False
    all_l1 = F.l1_loss(pred, target)
    non_input_l1 = F.l1_loss(pred[:, mask], target[:, mask])
    mse = F.mse_loss(pred[:, mask], target[:, mask])
    return 0.2 * all_l1 + 0.7 * non_input_l1 + 0.1 * mse


def lead_consistency_loss(pred: torch.Tensor, lead_names: List[str] = LEADS) -> torch.Tensor:
    residuals = lead_rule_residuals(pred, lead_names)
    if not residuals:
        return pred.new_tensor(0.0)
    return torch.stack([v.abs().mean() for v in residuals.values()]).mean()


def _axis_proxy(ecg: torch.Tensor, lead_names: List[str] = LEADS) -> torch.Tensor:
    idx = {lead: i for i, lead in enumerate(lead_names)}
    if "I" not in idx or "aVF" not in idx:
        return None
    energy = qrs_energy(ecg[:, [idx["I"], idx["aVF"]]])
    weights = energy / (energy.sum(dim=-1, keepdim=True) + 1e-6)
    lead_i = (ecg[:, 0] * weights[:, 0]).sum(dim=-1)
    lead_avf = (ecg[:, 5] * weights[:, 1]).sum(dim=-1)
    return F.normalize(torch.stack([lead_i, lead_avf], dim=-1), dim=-1, eps=1e-6)


def physiology_aux_loss(pred: torch.Tensor, target: torch.Tensor, lead_names: List[str] = LEADS) -> torch.Tensor:
    """Weak CDG-inspired losses, focused more on currently high-error leads."""
    lead_err = (pred.detach() - target.detach()).abs().mean(dim=-1)
    lead_w = lead_err / (lead_err.mean(dim=1, keepdim=True) + 1e-6)
    lead_w = lead_w.clamp(0.5, 2.0)

    pred_axis, target_axis = _axis_proxy(pred, lead_names), _axis_proxy(target, lead_names)
    axis = 1.0 - F.cosine_similarity(pred_axis, target_axis, dim=-1).mean() if pred_axis is not None else pred.new_tensor(0.0)

    pred_std = pred.std(dim=-1)
    target_std = target.std(dim=-1)
    amp = ((pred_std - target_std).abs() * lead_w).mean()

    idx = {lead: i for i, lead in enumerate(lead_names)}
    chest_terms = []
    for left, right in zip(("V1", "V2", "V3", "V4", "V5"), ("V2", "V3", "V4", "V5", "V6")):
        if left in idx and right in idx:
            pred_step = pred[:, idx[right]] - pred[:, idx[left]]
            target_step = target[:, idx[right]] - target[:, idx[left]]
            w = 0.5 * (lead_w[:, idx[right]] + lead_w[:, idx[left]]).unsqueeze(-1)
            chest_terms.append(((pred_step - target_step).abs() * w).mean())
    chest_transition = torch.stack(chest_terms).mean() if chest_terms else pred.new_tensor(0.0)

    return 0.4 * axis + 0.3 * amp + 0.3 * chest_transition


def physics_gate_target_loss(explain: Dict[str, torch.Tensor], pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gate = explain.get("physics_gate")
    if gate is None:
        return pred.new_tensor(0.0)
    unet_pred = explain.get("unet_pred", pred).detach()
    err_map = (unet_pred - target.detach()).abs()
    err_min = err_map.amin(dim=(1, 2), keepdim=True)
    err_max = err_map.amax(dim=(1, 2), keepdim=True)
    target_gate = ((err_map - err_min) / (err_max - err_min + 1e-6)).clamp(0.0, 1.0)
    with torch.cuda.amp.autocast(enabled=False):
        return F.binary_cross_entropy(gate.float().clamp(1e-4, 1.0 - 1e-4), target_gate.float())


def physics_residual_target_loss(explain: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    residual = explain.get("physics_residual")
    unet_pred = explain.get("unet_pred")
    if residual is None or unet_pred is None:
        return target.new_tensor(0.0)
    residual_target = target.detach() - unet_pred.detach()
    err_weight = residual_target.abs()
    err_weight = err_weight / (err_weight.mean(dim=(1, 2), keepdim=True) + 1e-6)
    err_weight = err_weight.clamp(0.25, 4.0)
    return (err_weight * (residual - residual_target).abs()).mean()


def explanation_sparsity_loss(explain: Dict[str, torch.Tensor]) -> torch.Tensor:
    mix = explain["lead_mix"]
    temporal = explain["temporal_importance"]
    residual_fraction = explain["residual_fraction"].mean()
    phys = explain.get("physics_residual")
    gate = explain.get("physics_gate")
    phys_reg = phys.abs().mean() if phys is not None else mix.new_tensor(0.0)
    gate_reg = gate.mean() if gate is not None else mix.new_tensor(0.0)
    return (
        0.01 * mix.abs().mean()
        + 0.01 * temporal.mean()
        + 0.05 * residual_fraction
        + 0.02 * phys_reg
        + 0.01 * gate_reg
    )


def qrs_energy(x: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
    diff = torch.diff(x, dim=-1, prepend=x[..., :1])
    weight = torch.ones(x.size(1), 1, kernel_size, device=x.device, dtype=x.dtype) / kernel_size
    energy = F.conv1d(diff.pow(2), weight, padding=kernel_size // 2, groups=x.size(1))
    return energy[..., : x.size(-1)]


def source_explanation_loss(explain: Dict[str, torch.Tensor], input_ecg: torch.Tensor) -> torch.Tensor:
    source = explain["source_activation"]
    source_norm = source / (source.amax(dim=-1, keepdim=True) + 1e-6)
    input_energy = qrs_energy(input_ecg)
    input_energy = input_energy / (input_energy.amax(dim=-1, keepdim=True) + 1e-6)

    source_global = source_norm.mean(dim=1, keepdim=True)
    qrs_align = F.l1_loss(source_global, input_energy)
    sparsity = source.mean()
    temporal_smooth = torch.diff(source, dim=-1).abs().mean()
    vector_smooth = torch.diff(explain["source_vector"], dim=-1).abs().mean()
    lead_contrib_sparse = explain["source_to_lead"].abs().mean()
    return 0.5 * qrs_align + 0.05 * sparsity + 0.1 * temporal_smooth + 0.05 * vector_smooth + 0.02 * lead_contrib_sparse


def describe_prediction(explain: Dict[str, torch.Tensor], sample: int = 0, top_k: int = 3) -> Dict[str, object]:
    """Compact numeric explanation suitable for logging or a report."""
    mix = explain["lead_mix"].detach().cpu()[sample]
    temporal = explain["temporal_importance"].detach().cpu()[sample, 0]
    lead_summary = {}
    for lead_idx, lead in enumerate(LEADS):
        vals, idx = mix[lead_idx].abs().topk(min(top_k, mix.size(1)))
        lead_summary[lead] = [(int(i), float(mix[lead_idx, i]), float(v)) for i, v in zip(idx, vals)]
    peak_times = temporal.topk(min(10, temporal.numel())).indices.sort()[0].tolist()
    source = explain["source_activation"].detach().cpu()[sample]
    source_strength = source.mean(dim=-1)
    source_vals, source_idx = source_strength.topk(min(top_k, source_strength.numel()))
    return {
        "top_basis_by_lead": lead_summary,
        "top_aha17_sources": [(int(i), float(v)) for i, v in zip(source_idx, source_vals)],
        "important_time_indices": [int(t) for t in peak_times],
        "mean_residual_fraction": float(explain["residual_fraction"].detach().cpu()[sample]),
    }
