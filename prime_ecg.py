"""PRIME-ECG: Prior-conditioned Reconstructor with Integrated Multi-stage ECG generation.

Standalone module — no project-internal imports required.

Architecture:
    past_12lead (B, 12, T) ─► PriorEncoder ─► prior (B, 64)
                                                        │
    lead_i      (B,  1, T) ─────────────────────────────┤
                                             CascadeECGRecon (FiLM-conditioned)
                                                        │
                                             recon_12lead (B, 12, T)

CascadeECGRecon stages (Einthoven arithmetic + residual UNets):
    Lead I  (input)
    Lead II  ← UNet(I)  + FiLM(prior)
    Lead III ← Einthoven(I, II) + residual UNet
    aVR/aVL/aVF ← Einthoven + residual UNet
    V1–V6   ← UNet(I, II, III, aVR, aVL, aVF) + FiLM(prior)

Usage:
    model = make_prime_ecg()
    model = load_prime_ecg("prime_ecg_best.pt", device="cuda")

    lead_i      = torch.randn(4, 1, 1000)   # z-score normalised per lead
    past_12lead = torch.randn(4, 12, 1000)  # z-score normalised per lead

    recon = model(lead_i, past_12lead=past_12lead)  # (4, 12, 1000)

Evaluation (PTB-XL):
    python prime_ecg.py eval \\
        --data_dir /path/to/ptb-xl \\
        --ckpt    prime_ecg_best.pt \\
        --device  cuda
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Lead order & index map ────────────────────────────────────────────────────

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
IDX   = {l: i for i, l in enumerate(LEADS)}

# ── UNet building blocks ──────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ic, oc, 7, padding=3), nn.BatchNorm1d(oc), nn.GELU(),
            nn.Conv1d(oc, oc, 5, padding=2), nn.BatchNorm1d(oc), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, ic: int, oc: int):
        super().__init__()
        self.down  = nn.Conv1d(ic, oc, 4, stride=2, padding=1)
        self.block = ConvBlock(oc, oc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, ic: int, sc: int, oc: int):
        super().__init__()
        self.up    = nn.ConvTranspose1d(ic, oc, 4, stride=2, padding=1)
        self.block = ConvBlock(oc + sc, oc)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNet1D(nn.Module):
    """1-D UNet with optional FiLM conditioning at the bottleneck."""

    def __init__(self, in_ch: int, base_ch: int = 32, out_ch: int = 1, film_dim: int = 0):
        super().__init__()
        b = base_ch
        self.enc1 = ConvBlock(in_ch, b)
        self.enc2 = DownBlock(b,     b * 2)
        self.enc3 = DownBlock(b * 2, b * 4)
        self.enc4 = DownBlock(b * 4, b * 8)
        self.bot  = ConvBlock(b * 8, b * 8)
        self.up3  = UpBlock(b * 8, b * 4, b * 4)
        self.up2  = UpBlock(b * 4, b * 2, b * 2)
        self.up1  = UpBlock(b * 2, b,     b)
        self.head = nn.Conv1d(b, out_ch, 1)
        self.film = nn.Linear(film_dim, b * 8 * 2) if film_dim > 0 else None

    def forward(self, x: torch.Tensor, meta_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bot(e4)
        if self.film is not None and meta_h is not None:
            gamma, beta = self.film(meta_h).chunk(2, dim=-1)
            b = b * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        d3 = self.up3(b,  e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)

# ── Cascade ECG reconstruction model ─────────────────────────────────────────

class CascadeECGRecon(nn.Module):
    """Physics-constrained cascade: I → II → limb (Einthoven + residual) → V leads.

    Args:
        base_ch:        UNet base channels for gen_II and gen_V (default 32)
        res_base_ch:    UNet base channels for residual limb UNets (default 16)
        residual_scale: tanh output scale for residual corrections (default 0.3)
        meta_dim:       FiLM conditioning dimension; 0 = no conditioning (default 0)
    """

    def __init__(
        self,
        base_ch:        int   = 32,
        res_base_ch:    int   = 16,
        residual_scale: float = 0.3,
        meta_dim:       int   = 0,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.meta_dim       = meta_dim
        film_d              = 64 if meta_dim > 0 else 0

        self.meta_enc = nn.Sequential(
            nn.Linear(meta_dim, 32), nn.GELU(),
            nn.Linear(32, 64),
        ) if meta_dim > 0 else None

        # Stage 1: I → II
        self.gen_II  = UNet1D(1, base_ch,     1, film_dim=film_d)
        # Stage 2: (I, II) → limb residuals (Einthoven arithmetic + small correction)
        self.res_III = UNet1D(2, res_base_ch, 1)
        self.res_aVR = UNet1D(2, res_base_ch, 1)
        self.res_aVL = UNet1D(2, res_base_ch, 1)
        self.res_aVF = UNet1D(2, res_base_ch, 1)
        # Stage 3: (I, II, III, aVR, aVL, aVF) → V1-V6
        self.gen_V   = UNet1D(6, base_ch,     6, film_dim=film_d)

    def forward(
        self,
        x_I:              torch.Tensor,
        gt_full:          Optional[torch.Tensor] = None,
        teacher_forcing:  float                  = 0.0,
        meta:             Optional[torch.Tensor] = None,
        return_intermediate: bool                = False,
    ):
        meta_h = self.meta_enc(meta) if (self.meta_enc is not None and meta is not None) else None

        # Stage 1 ─────────────────────────────────────────────────────────────
        pred_II = self.gen_II(x_I, meta_h=meta_h)

        use_gt = (gt_full is not None) and (torch.rand(1).item() < teacher_forcing)
        ctx_II = gt_full[:, IDX["II"]:IDX["II"]+1] if use_gt else pred_II

        # Stage 2 (Einthoven arithmetic + learned residual) ───────────────────
        ctx2     = torch.cat([x_I, ctx_II], dim=1)
        arith_III = ctx_II - x_I
        arith_aVR = -(x_I + ctx_II) / 2.0
        arith_aVL = x_I  - ctx_II   / 2.0
        arith_aVF = ctx_II - x_I    / 2.0

        s = self.residual_scale
        pred_III = arith_III + s * torch.tanh(self.res_III(ctx2))
        pred_aVR = arith_aVR + s * torch.tanh(self.res_aVR(ctx2))
        pred_aVL = arith_aVL + s * torch.tanh(self.res_aVL(ctx2))
        pred_aVF = arith_aVF + s * torch.tanh(self.res_aVF(ctx2))

        # Stage 3 ─────────────────────────────────────────────────────────────
        if use_gt and gt_full is not None:
            ctx3 = torch.cat([
                x_I,
                gt_full[:, IDX["II"]:IDX["II"]+1],
                gt_full[:, IDX["III"]:IDX["III"]+1],
                gt_full[:, IDX["aVR"]:IDX["aVR"]+1],
                gt_full[:, IDX["aVL"]:IDX["aVL"]+1],
                gt_full[:, IDX["aVF"]:IDX["aVF"]+1],
            ], dim=1)
        else:
            ctx3 = torch.cat([x_I, pred_II, pred_III, pred_aVR, pred_aVL, pred_aVF], dim=1)
        pred_V = self.gen_V(ctx3, meta_h=meta_h)

        out = torch.cat([x_I, pred_II, pred_III, pred_aVR, pred_aVL, pred_aVF, pred_V], dim=1)

        if not return_intermediate:
            return out
        return out, {
            "pred_II": pred_II,
            "pred_III": pred_III, "pred_aVR": pred_aVR,
            "pred_aVL": pred_aVL, "pred_aVF": pred_aVF,
            "pred_V":  pred_V,
            "tf_used": use_gt,
        }

# ── Prior encoder ─────────────────────────────────────────────────────────────

class PriorEncoder(nn.Module):
    """Encodes a past 12-lead ECG into a fixed-size prior vector.

    Args:
        in_ch:          number of input leads (default 12)
        out_dim:        output dimension (default 64)
        use_time_delta: if True, reserves 1 dim for time_delta (default True)
    """

    def __init__(self, in_ch: int = 12, out_dim: int = 64, use_time_delta: bool = True):
        super().__init__()
        self.use_time_delta = use_time_delta
        ecg_dim = (out_dim - 1) if use_time_delta else out_dim
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 15, padding=7, stride=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64,  7, padding=3, stride=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, ecg_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ── Prior-conditioned wrapper ─────────────────────────────────────────────────

class PriorConditionedRecon(nn.Module):
    """Wraps any ECG reconstructor with prior conditioning from a past 12-lead ECG.

    Forward args:
        lead_i      : (B, 1,  T) — current Lead I, z-score normalised
        past_12lead : (B, 12, T) — previous 12-lead ECG, z-score normalised
        time_delta  : (B, 1)     — days since past recording (optional)
        patient_meta: (B, M)     — demographic features (optional, when use_meta=True)
        gt_full     : (B, 12, T) — ground truth for teacher forcing (training only)
        teacher_forcing : float  — probability of using ground truth context [0, 1]

    Returns:
        recon : (B, 12, T)
        -- or --
        (recon, intermediates_dict)  when return_intermediate=True
    """

    def __init__(
        self,
        base_model:     nn.Module,
        prior_dim:      int  = 64,
        use_meta:       bool = False,
        use_time_delta: bool = True,
    ):
        super().__init__()
        self.base           = base_model
        self.use_time_delta = use_time_delta
        self.use_meta       = use_meta
        self.prior_enc      = PriorEncoder(in_ch=12, out_dim=prior_dim,
                                           use_time_delta=use_time_delta)

    def forward(
        self,
        lead_i:          torch.Tensor,
        past_12lead:     Optional[torch.Tensor] = None,
        time_delta:      Optional[torch.Tensor] = None,
        patient_meta:    Optional[torch.Tensor] = None,
        gt_full:         Optional[torch.Tensor] = None,
        teacher_forcing: float                  = 0.0,
        return_intermediate: bool               = False,
    ):
        parts = []
        if past_12lead is not None:
            ecg_feat = self.prior_enc(past_12lead)
            if self.use_time_delta:
                if time_delta is None:
                    time_delta = torch.zeros(ecg_feat.size(0), 1, device=ecg_feat.device)
                parts.append(torch.cat([ecg_feat, time_delta], dim=-1))
            else:
                parts.append(ecg_feat)
        if self.use_meta and patient_meta is not None:
            parts.append(patient_meta)
        conditioning = torch.cat(parts, dim=-1) if parts else None
        return self.base(
            lead_i,
            gt_full=gt_full,
            teacher_forcing=teacher_forcing,
            meta=conditioning,
            return_intermediate=return_intermediate,
        )

# ── Factory & checkpoint helpers ──────────────────────────────────────────────

def make_prime_ecg(
    prior_dim:   int   = 64,
    base_ch:     int   = 32,
    res_base_ch: int   = 16,
) -> PriorConditionedRecon:
    """Build a PRIME-ECG model with default hyperparameters.

    Args:
        prior_dim:   dimension of the prior ECG embedding (default 64)
        base_ch:     UNet base channels for gen_II / gen_V (default 32)
        res_base_ch: UNet base channels for limb residual nets (default 16)

    Returns:
        PriorConditionedRecon wrapping CascadeECGRecon
    """
    base = CascadeECGRecon(
        base_ch=base_ch,
        res_base_ch=res_base_ch,
        meta_dim=prior_dim,
    )
    return PriorConditionedRecon(base, prior_dim=prior_dim, use_meta=False, use_time_delta=False)


def load_prime_ecg(
    ckpt_path: str,
    device:    str = "cpu",
    prior_dim: int = 64,
    base_ch:   int = 32,
    res_base_ch: int = 16,
) -> PriorConditionedRecon:
    """Load a PRIME-ECG model from a checkpoint file.

    Args:
        ckpt_path:   path to the .pt checkpoint (state_dict)
        device:      torch device string (default 'cpu')
        prior_dim:   must match the value used at training time (default 64)
        base_ch:     must match the value used at training time (default 32)
        res_base_ch: must match the value used at training time (default 16)

    Returns:
        PriorConditionedRecon loaded with trained weights, set to eval mode
    """
    model = make_prime_ecg(prior_dim=prior_dim, base_ch=base_ch, res_base_ch=res_base_ch)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


# ── PTB-XL evaluation helpers ─────────────────────────────────────────────────

def _normalize_ecg(sig: np.ndarray) -> np.ndarray:
    """Global z-score normalization: (T, 12) → (12, T).
    Mean and std computed over all leads and all time points (scalar).
    """
    sig = sig.astype(np.float32).T          # (12, T)
    return (sig - sig.mean()) / (sig.std() + 1e-6)


def _read_ecg(data_dir, rec_path: str, crop: int = 1000) -> np.ndarray:
    import wfdb
    sig, _ = wfdb.rdsamp(str(data_dir / rec_path))
    ecg = _normalize_ecg(sig)              # (12, T)
    return ecg[:, :crop]                   # first 1000 samples (500 Hz → 2 s)


def _time_delta_days(a, b) -> float:
    import pandas as pd
    da, db = pd.to_datetime(a, errors="coerce"), pd.to_datetime(b, errors="coerce")
    if pd.isna(da) or pd.isna(db):
        return 0.0
    return max(0.0, float((db - da).days))


def _build_test_pairs(data_dir, crop: int = 1000):
    """Build consecutive-pair test set from PTB-XL fold 10 (294 pairs)."""
    import pandas as pd
    df = pd.read_csv(str(data_dir / "ptbxl_database.csv"))
    df["recording_date"] = pd.to_datetime(df["recording_date"], errors="coerce")
    fold10 = df[df["strat_fold"] == 10].copy()
    fold10 = fold10.sort_values(["patient_id", "recording_date", "ecg_id"])

    fname_col = "filename_hr" if crop > 500 else "filename_lr"
    if fname_col not in fold10.columns:
        fname_col = "filename_hr"

    pairs = []
    for pid, grp in fold10.groupby("patient_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue
        for j in range(1, len(grp)):
            past_row, cur_row = grp.iloc[j - 1], grp.iloc[j]
            pairs.append({
                "patient_id": pid,
                "past":       past_row[fname_col],
                "current":    cur_row[fname_col],
                "delta_days": _time_delta_days(past_row["recording_date"], cur_row["recording_date"]),
            })
    return pairs


def evaluate_ptbxl(
    ckpt_path: str,
    data_dir: str,
    device: str = "cuda",
    batch_size: int = 32,
    crop: int = 1000,
    prior_dim: int = 64,
    base_ch: int = 32,
    res_base_ch: int = 16,
) -> dict:
    """Evaluate PRIME-ECG on PTB-XL fold-10 test set (294 consecutive pairs).

    Args:
        ckpt_path:   path to prime_ecg_best.pt
        data_dir:    PTB-XL root (contains ptbxl_database.csv + records500/)
        device:      'cuda' or 'cpu'
        batch_size:  inference batch size
        crop:        number of time samples to use (default 1000 = 2 s at 500 Hz)

    Returns:
        dict with keys pcc_mean, mse_mean, and per-lead mse_<lead> / pcc_<lead>
    """
    from pathlib import Path
    from torch.utils.data import DataLoader, Dataset

    data_dir = Path(data_dir)
    model = load_prime_ecg(ckpt_path, device=device,
                           prior_dim=prior_dim, base_ch=base_ch, res_base_ch=res_base_ch)

    pairs = _build_test_pairs(data_dir, crop=crop)
    print(f"[eval] test pairs: {len(pairs)}")

    class _DS(Dataset):
        def __init__(self, pairs, data_dir, crop):
            self.pairs, self.data_dir, self.crop = pairs, data_dir, crop
        def __len__(self):  return len(self.pairs)
        def __getitem__(self, i):
            p = self.pairs[i]
            cur  = _read_ecg(self.data_dir, p["current"], self.crop)
            past = _read_ecg(self.data_dir, p["past"],    self.crop)
            td   = np.log1p(p["delta_days"]) / np.log1p(3650.0)
            return {
                "lead_i":     torch.from_numpy(cur[IDX["I"]:IDX["I"]+1]),
                "target":     torch.from_numpy(cur),
                "past":       torch.from_numpy(past),
                "time_delta": torch.tensor([td], dtype=torch.float32),
            }

    loader = DataLoader(_DS(pairs, data_dir, crop),
                        batch_size=batch_size, shuffle=False, num_workers=4)

    all_pred, all_gt = [], []
    model.eval()
    dev = torch.device(device)
    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["lead_i"].to(dev),
                past_12lead=batch["past"].to(dev),
            )                                       # (B, 12, T)
            all_pred.append(pred.cpu())
            all_gt.append(batch["target"])

    pred_all = torch.cat(all_pred, dim=0)           # (N, 12, T)
    gt_all   = torch.cat(all_gt,   dim=0)           # (N, 12, T)

    results = {}
    mses, pccs = [], []
    for li, lead in enumerate(LEADS):
        p = pred_all[:, li]                         # (N, T)
        g = gt_all[:,  li]
        mse = float(((p - g) ** 2).mean())
        # Pearson per sample, then average
        p_z = p - p.mean(dim=1, keepdim=True)
        g_z = g - g.mean(dim=1, keepdim=True)
        num = (p_z * g_z).sum(dim=1)
        den = (p_z.norm(dim=1) * g_z.norm(dim=1)).clamp(min=1e-8)
        pcc = float((num / den).mean())
        results[f"mse_{lead}"] = mse
        results[f"pcc_{lead}"] = pcc
        mses.append(mse)
        pccs.append(pcc)

    results["mse_mean"] = float(np.mean(mses))
    results["pcc_mean"] = float(np.mean(pccs))
    return results


# ── Smoke test / CLI evaluation ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        import argparse
        ap = argparse.ArgumentParser(description="Evaluate PRIME-ECG on PTB-XL fold-10")
        ap.add_argument("--data_dir", required=True, help="PTB-XL root directory")
        ap.add_argument("--ckpt",     required=True, help="path to prime_ecg_best.pt")
        ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
        ap.add_argument("--batch_size", type=int, default=32)
        ap.add_argument("--crop",     type=int, default=1000,
                        help="ECG samples to use (default 1000 = 2s @ 500Hz)")
        args = ap.parse_args(sys.argv[2:])

        res = evaluate_ptbxl(
            ckpt_path=args.ckpt,
            data_dir=args.data_dir,
            device=args.device,
            batch_size=args.batch_size,
            crop=args.crop,
        )
        print(f"\n{'Lead':<6} {'MSE':>8} {'PCC':>8}")
        print("-" * 26)
        for lead in LEADS:
            print(f"{lead:<6} {res[f'mse_{lead}']:8.4f} {res[f'pcc_{lead}']:8.4f}")
        print("-" * 26)
        print(f"{'mean':<6} {res['mse_mean']:8.4f} {res['pcc_mean']:8.4f}")
    else:
        # Smoke test
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"device: {device}")

        model = make_prime_ecg().to(device)
        total = sum(p.numel() for p in model.parameters())
        print(f"parameters: {total:,}")

        B, T = 4, 1000
        lead_i      = torch.randn(B, 1,  T, device=device)
        past_12lead = torch.randn(B, 12, T, device=device)

        with torch.no_grad():
            recon = model(lead_i, past_12lead=past_12lead)

        print(f"input  lead_i      : {tuple(lead_i.shape)}")
        print(f"input  past_12lead : {tuple(past_12lead.shape)}")
        print(f"output recon       : {tuple(recon.shape)}")

        assert torch.allclose(recon[:, IDX["I"]:IDX["I"]+1], lead_i, atol=1e-5), \
            "Lead I not preserved in output!"
        print("Lead I preserved: OK")
        print("Smoke test passed.")
