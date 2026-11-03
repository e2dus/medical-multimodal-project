# -*- coding: utf-8 -*-
"""
PRIME-ECG full experiment driver.

This script runs the comparisons needed to defend a longitudinal
Lead-I + prior-12-lead ECG reconstruction paper:

1. Reconstruction baselines
   - past_as_output
   - lead_i
   - ecgrecover
   - ecgnet
   - ekgan
   - mcma
   - lead_i_meta
   - concat
   - prime_ecg
   - prime_ecg_meta
   - no_physics

2. PRIME-ECG ablations
   - self prior
   - random prior
   - demographic-matched prior
   - zero prior
   - zero time_delta
   - zero metadata

3. Temporal gap robustness
   - 0-1 day, 1-7 days, 7-30 days, 1-6 months, 6-12 months, 1 year+

4. Disease-specific lead selection
   - real 12-lead
   - Lead I only
   - past 12-lead
   - reconstructed all leads
   - reconstructed selected leads
   - reconstructed random same-number leads

Assumptions:
    - PTB-XL style folder with ptbxl_database.csv and WFDB records.
    - cascade_recon.py is available on the server.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        roc_auc_score,
    )
except Exception:
    average_precision_score = None
    f1_score = None
    roc_auc_score = None

from cascade_recon import (
    CascadeECGRecon,
    VanillaECGRecon,
    LEADS,
    IDX,
    cascade_loss,
    get_lead_weight_tensor,
)


# ---------------------------------------------------------------------------
# Optional official baseline repos.
# The local desktop workspace uses ecgrecover_official / ekgan_official folders,
# while the training server may keep clones under /workspace.  Environment
# variables win when set.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent


def _first_existing_path(env_name: str, candidates: Sequence[Path]) -> str:
    env_path = os.environ.get(env_name, "").strip()
    if env_path:
        return env_path
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0]) if candidates else ""


ECGRECOVER_REPO = _first_existing_path(
    "ECGRECOVER_REPO",
    [
        _PROJECT_ROOT / "ecgrecover",
        _PROJECT_ROOT / "ecgrecover_official",
        _PROJECT_ROOT / "ecg-recon" / "ecgrecover",
        Path("/workspace/ecgrecover"),
    ],
)

EKGAN_REPO = _first_existing_path(
    "EKGAN_REPO",
    [
        _PROJECT_ROOT / "ekgan",
        _PROJECT_ROOT / "ekgan_official",
        _PROJECT_ROOT / "ecg-recon" / "ekgan",
        Path("/workspace/ekgan"),
    ],
)

_OfficialECGRecoverNet = None
try:
    import sys as _sys
    if ECGRECOVER_REPO and ECGRECOVER_REPO not in _sys.path:
        _sys.path.insert(0, ECGRECOVER_REPO)
    from tools.LoadModel import Autoencoder_net as _OfficialECGRecoverNet  # type: ignore
except Exception as _e:  # noqa: BLE001
    _OfficialECGRecoverNet = None
    print(f"[info] official ECGrecover not available ({_e}); "
          f"only 'ecgrecover' (re-impl) usable, not 'ecgrecover_official'.")


GAP_BINS = [
    ("0-1d", 0.0, 1.0),
    ("1-7d", 1.0, 7.0),
    ("7-30d", 7.0, 30.0),
    ("1-6m", 30.0, 183.0),
    ("6-12m", 183.0, 365.0),
    ("1y+", 365.0, float("inf")),
]

LIMB_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]
ANTERIOR_LEADS = ["V1", "V2", "V3", "V4"]
CHEST_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class Args:
    data_dir: str = "/workspace/physionet.org/files/ptb-xl/1.0.3/"
    output_dir: str = "./outputs/prime_ecg_full_exp"
    use_hr: bool = False
    epochs: int = 30
    classifier_epochs: int = 15
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    base_ch: int = 32
    official_signal_len: int = 512
    ekgan_lambda: float = 100.0
    ekgan_alpha: float = 1.0
    ekgan_lr_decay: float = 0.95
    ekgan_decay_start_epoch: int = 5
    res_base_ch: int = 16
    residual_scale: float = 0.3
    prior_dim: int = 64
    tf_decay_ratio: float = 0.5
    teacher_forcing: bool = False
    train_folds: str = "1,2,3,4,5,6,7,8"
    val_folds: str = "9"
    test_folds: str = "10"
    methods: str = "lead_i,ecgrecover,ecgrecover_official,ecgnet,ekgan,ekgan_prior,mcma,mcma_prior,mcma_prior_enc,lead_i_meta,concat,prime_ecg,prime_ecg_meta,no_physics"
    run_classification: bool = False
    disease_labels: str = "NORM,MI,STTC,CD,HYP"
    select_top_k: int = 4
    classifier_lead_dropout: float = 0.25
    max_train_pairs: int = 0
    max_val_pairs: int = 0
    max_test_pairs: int = 0
    amp: bool = True


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_literal_dict(x: object) -> Dict[str, float]:
    if isinstance(x, dict):
        return x
    try:
        out = ast.literal_eval(str(x))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def load_scp_mapping(data_dir: Path) -> Dict[str, str]:
    path = data_dir / "scp_statements.csv"
    if not path.exists():
        return {}
    scp = pd.read_csv(path, index_col=0)
    mapping = {}
    for code, row in scp.iterrows():
        if bool(row.get("diagnostic", False)):
            mapping[str(code)] = str(row.get("diagnostic_class", ""))
    return mapping


def add_labels(df: pd.DataFrame, data_dir: Path, labels: Sequence[str]) -> pd.DataFrame:
    scp_map = load_scp_mapping(data_dir)
    out = df.copy()
    for label in labels:
        vals = []
        for item in out["scp_codes"].tolist():
            codes = safe_literal_dict(item)
            hit = False
            for code, score in codes.items():
                if float(score) <= 0:
                    continue
                if str(code) == label or scp_map.get(str(code), "") == label:
                    hit = True
                    break
            vals.append(1.0 if hit else 0.0)
        out[f"label_{label}"] = vals
    return out


def normalize_ecg(sig: np.ndarray) -> np.ndarray:
    sig = sig.astype(np.float32).T
    return (sig - sig.mean()) / (sig.std() + 1e-6)


def read_ecg(data_dir: Path, rec_path: str) -> np.ndarray:
    sig, _ = wfdb.rdsamp(str(data_dir / rec_path))
    return normalize_ecg(sig)


def time_delta_days(a: object, b: object) -> float:
    da = pd.to_datetime(a, errors="coerce")
    db = pd.to_datetime(b, errors="coerce")
    if pd.isna(da) or pd.isna(db):
        return 0.0
    return max(0.0, float((db - da).days))


def limit_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n and len(df) > n:
        return df.iloc[:n].reset_index(drop=True)
    return df.reset_index(drop=True)


def build_pairs(df: pd.DataFrame, fname_col: str, fold_values: Sequence[int], labels: Sequence[str]) -> pd.DataFrame:
    sub = df[df["strat_fold"].isin(fold_values)].copy()
    if "recording_date" in sub.columns:
        sub["recording_date"] = pd.to_datetime(sub["recording_date"], errors="coerce")
        sub = sub.sort_values(["patient_id", "recording_date", "ecg_id"])
    else:
        sub["recording_date"] = pd.NaT
        sub = sub.sort_values(["patient_id", "ecg_id"])

    rows = []
    for pid, grp in sub.groupby("patient_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < 2:
            continue
        for j in range(1, len(grp)):
            past = grp.iloc[j - 1]
            cur = grp.iloc[j]
            row = {
                "patient_id": pid,
                "past": past[fname_col],
                "current": cur[fname_col],
                "delta_days": time_delta_days(past["recording_date"], cur["recording_date"]),
                "age": cur.get("age", np.nan),
                "sex": cur.get("sex", np.nan),
                "height": cur.get("height", np.nan),
                "weight": cur.get("weight", np.nan),
            }
            for label in labels:
                row[f"label_{label}"] = cur.get(f"label_{label}", 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def assert_no_patient_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    tr = set(train["patient_id"].astype(str))
    va = set(val["patient_id"].astype(str))
    te = set(test["patient_id"].astype(str))
    if tr & va or tr & te or va & te:
        raise RuntimeError("Patient overlap detected across train/val/test splits.")


def compute_meta_stats(train_pairs: pd.DataFrame) -> Dict[str, float]:
    h = train_pairs["height"].dropna()
    w = train_pairs["weight"].dropna()
    return {
        "h_mean": float(h.mean()) if len(h) else 170.0,
        "h_std": float(h.std()) if len(h) > 1 else 10.0,
        "w_mean": float(w.mean()) if len(w) else 75.0,
        "w_std": float(w.std()) if len(w) > 1 else 15.0,
    }


def meta_vector(row: pd.Series, stats: Dict[str, float]) -> np.ndarray:
    age = float(row.get("age", 60.0)) if not pd.isna(row.get("age", np.nan)) else 60.0
    sex_raw = str(row.get("sex", "0"))
    sex = 1.0 if sex_raw in ("1", "1.0", "F", "female", "Female") else 0.0
    h_raw = row.get("height", np.nan)
    w_raw = row.get("weight", np.nan)
    h_missing = 1.0 if pd.isna(h_raw) else 0.0
    w_missing = 1.0 if pd.isna(w_raw) else 0.0
    height = stats["h_mean"] if h_missing else float(h_raw)
    weight = stats["w_mean"] if w_missing else float(w_raw)
    return np.array(
        [
            age / 100.0,
            sex,
            (height - stats["h_mean"]) / (stats["h_std"] + 1e-6),
            (weight - stats["w_mean"]) / (stats["w_std"] + 1e-6),
            h_missing,
            w_missing,
        ],
        dtype=np.float32,
    )


def demo_key(row: pd.Series) -> Tuple[str, str, str]:
    age = row.get("age", np.nan)
    age_group = "age_na" if pd.isna(age) else f"age_{int(float(age) // 10) * 10}"
    sex = str(row.get("sex", "sex_na"))
    height = row.get("height", np.nan)
    weight = row.get("weight", np.nan)
    if pd.isna(height) or pd.isna(weight) or float(height) <= 0:
        bmi_group = "bmi_na"
    else:
        h_m = float(height) / 100.0 if float(height) > 3 else float(height)
        bmi = float(weight) / (h_m * h_m + 1e-6)
        bmi_group = f"bmi_{int(bmi // 5) * 5}"
    return sex, age_group, bmi_group


class PriorPairDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        pairs: pd.DataFrame,
        label_names: Sequence[str],
        meta_stats: Dict[str, float],
        pool_pairs: Optional[pd.DataFrame] = None,
        past_mode: str = "self",
        zero_time: bool = False,
        zero_meta: bool = False,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.pairs = pairs.reset_index(drop=True)
        self.label_names = list(label_names)
        self.meta_stats = meta_stats
        self.pool_pairs = pool_pairs.reset_index(drop=True) if pool_pairs is not None else self.pairs
        self.past_mode = past_mode
        self.zero_time = zero_time
        self.zero_meta = zero_meta
        self.rng = np.random.default_rng(seed)
        self.demo_index: Dict[Tuple[str, str, str], List[int]] = {}
        for i, row in self.pool_pairs.iterrows():
            self.demo_index.setdefault(demo_key(row), []).append(i)

    def __len__(self) -> int:
        return len(self.pairs)

    def _other_past_path(self, row: pd.Series, demographic: bool) -> str:
        if demographic:
            candidates = self.demo_index.get(demo_key(row), [])
        else:
            candidates = list(range(len(self.pool_pairs)))
        if not candidates:
            candidates = list(range(len(self.pool_pairs)))
        pid = str(row["patient_id"])
        for _ in range(50):
            idx = int(self.rng.choice(candidates))
            cand = self.pool_pairs.iloc[idx]
            if str(cand["patient_id"]) != pid:
                return str(cand["past"])
        return str(self.pool_pairs.iloc[int(self.rng.integers(0, len(self.pool_pairs)))]["past"])

    def __getitem__(self, i: int):
        row = self.pairs.iloc[i]
        cur = read_ecg(self.data_dir, str(row["current"]))
        if self.past_mode == "self":
            past = read_ecg(self.data_dir, str(row["past"]))
        elif self.past_mode == "random":
            past = read_ecg(self.data_dir, self._other_past_path(row, demographic=False))
        elif self.past_mode == "demo":
            past = read_ecg(self.data_dir, self._other_past_path(row, demographic=True))
        elif self.past_mode == "zero":
            past = np.zeros_like(cur, dtype=np.float32)
        else:
            raise ValueError(f"Unknown past_mode={self.past_mode}")

        td = 0.0 if self.zero_time else float(row["delta_days"])
        td = np.log1p(td) / np.log1p(3650.0)
        meta = np.zeros(6, dtype=np.float32) if self.zero_meta else meta_vector(row, self.meta_stats)
        labels = np.array([float(row.get(f"label_{x}", 0.0)) for x in self.label_names], dtype=np.float32)
        return {
            "lead_i": torch.from_numpy(cur[IDX["I"] : IDX["I"] + 1]),
            "target": torch.from_numpy(cur),
            "past": torch.from_numpy(past),
            "time_delta": torch.tensor([td], dtype=torch.float32),
            "meta": torch.from_numpy(meta),
            "labels": torch.from_numpy(labels),
            "delta_days": torch.tensor(float(row["delta_days"]), dtype=torch.float32),
        }


class PriorEncoder(nn.Module):
    def __init__(self, in_ch: int = 12, out_dim: int = 64):
        super().__init__()
        if out_dim < 2:
            raise ValueError("prior_dim must be >= 2 because one dimension is reserved for time_delta.")
        ecg_dim = out_dim - 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 15, padding=7, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, ecg_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PriorConditionedRecon(nn.Module):
    def __init__(self, base_model: nn.Module, prior_dim: int = 64, use_meta: bool = False):
        super().__init__()
        self.base = base_model
        self.prior_enc = PriorEncoder(in_ch=12, out_dim=prior_dim)
        self.use_meta = use_meta

    def forward(
        self,
        lead_i: torch.Tensor,
        past_12lead: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        patient_meta: Optional[torch.Tensor] = None,
        gt_full: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_intermediate: bool = False,
    ):
        parts = []
        if past_12lead is not None:
            ecg_feat = self.prior_enc(past_12lead)
            if time_delta is None:
                time_delta = torch.zeros(ecg_feat.size(0), 1, device=ecg_feat.device)
            parts.append(torch.cat([ecg_feat, time_delta], dim=-1))
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


class LeadIOnlyRecon(nn.Module):
    def __init__(self, base_model: nn.Module, use_meta: bool = False):
        super().__init__()
        self.base = base_model
        self.use_meta = use_meta

    def forward(
        self,
        lead_i: torch.Tensor,
        past_12lead: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        patient_meta: Optional[torch.Tensor] = None,
        gt_full: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_intermediate: bool = False,
    ):
        meta = patient_meta if self.use_meta else None
        return self.base(
            lead_i,
            gt_full=gt_full,
            teacher_forcing=teacher_forcing,
            meta=meta,
            return_intermediate=return_intermediate,
        )


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


class ConcatUNetRecon(nn.Module):
    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(14, base_ch)
        self.down1 = nn.Conv1d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.enc3 = ConvBlock(base_ch * 4, base_ch * 4)
        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out = nn.Conv1d(base_ch, 12, 1)

    def forward(
        self,
        lead_i: torch.Tensor,
        past_12lead: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        patient_meta: Optional[torch.Tensor] = None,
        gt_full: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_intermediate: bool = False,
    ):
        if past_12lead is None:
            past_12lead = torch.zeros(lead_i.size(0), 12, lead_i.size(-1), device=lead_i.device)
        if time_delta is None:
            time_delta = torch.zeros(lead_i.size(0), 1, device=lead_i.device)
        td_ch = time_delta.view(lead_i.size(0), 1, 1).expand(lead_i.size(0), 1, lead_i.size(-1)).to(dtype=lead_i.dtype)
        x = torch.cat([lead_i, past_12lead, td_ch], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        d2 = self.up2(e3)
        if d2.size(-1) != e2.size(-1):
            d2 = F.interpolate(d2, size=e2.size(-1), mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.size(-1) != e1.size(-1):
            d1 = F.interpolate(d1, size=e1.size(-1), mode="linear", align_corners=False)
        pred = self.out(self.dec1(torch.cat([d1, e1], dim=1)))
        if return_intermediate:
            return pred, {}
        return pred


class OfficialECGRecover(nn.Module):
    """Wrapper around the OFFICIAL UMMISCO/ecgrecover Autoencoder_net.

    The official model expects input shape (B, 1, 12, T) where unobserved
    leads are zero-padded; it outputs (B, 12, T) (after the wrapper's squeeze
    in the original forward).  We feed only Lead I and zero out the other 11
    channels, matching how Lead-I-only reconstruction is evaluated.

    NOTE: This loads the *original* PyTorch architecture from
    ``$ECGRECOVER_REPO/tools/LoadModel.py``.  Set ``ECGRECOVER_REPO`` at the
    top of this file.  If the import fails, ``make_model('ecgrecover_official')``
    will raise.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        load_pretrained: bool = True,
        signal_len: int = 512,
    ):
        """
        ``load_pretrained=True`` (default) auto-loads the official Model.pth
        from ``ECGRECOVER_REPO/model/Model.pth`` if present, so we start from
        the authors' released checkpoint and fine-tune on PTB-XL. This gives
        ECGrecover a strong, paper-aligned baseline.
        """
        super().__init__()
        self.signal_len = int(signal_len)
        if _OfficialECGRecoverNet is None:
            raise RuntimeError(
                "Official ECGrecover repo not importable. "
                f"Check ECGRECOVER_REPO={ECGRECOVER_REPO}"
            )
        self.net = _OfficialECGRecoverNet()
        if load_pretrained and pretrained_path is None:
            candidate = Path(ECGRECOVER_REPO) / "model" / "Model.pth"
            if candidate.exists():
                pretrained_path = str(candidate)
        if pretrained_path and Path(pretrained_path).exists():
            try:
                sd = torch.load(pretrained_path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                missing, unexpected = self.net.load_state_dict(sd, strict=False)
                print(f"[info] Loaded official ECGrecover weights from {pretrained_path}")
                if missing:
                    print(f"       (missing keys: {len(missing)})")
                if unexpected:
                    print(f"       (unexpected keys: {len(unexpected)})")
            except Exception as _e:
                print(f"[warn] Could not load ECGrecover weights: {_e}")

    def forward(
        self,
        lead_i: torch.Tensor,
        past_12lead: Optional[torch.Tensor] = None,  # unused (single-shot baseline)
        time_delta: Optional[torch.Tensor] = None,
        patient_meta: Optional[torch.Tensor] = None,
        gt_full: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_intermediate: bool = False,
    ):
        # lead_i: (B, 1, T)  ->  pack into a 12-lead "image" with only Lead I filled.
        # Official ECGrecover expects input shape (B, 1, 12, T) — leads as rows,
        # time as columns — and returns (B, 12, T) after an internal squeeze.
        b, _, t = lead_i.shape
        device = lead_i.device
        lead_resized = lead_i
        if t != self.signal_len:
            lead_resized = F.interpolate(
                lead_i.float(), size=self.signal_len, mode="linear", align_corners=False
            ).to(dtype=lead_i.dtype)
        twelve = torch.zeros(b, 12, self.signal_len, device=device, dtype=lead_i.dtype)
        twelve[:, IDX["I"] : IDX["I"] + 1] = lead_resized
        x = twelve.unsqueeze(1).contiguous()  # (B, 1, 12, 512)
        out = self.net(x, device)              # expected: (B, 12, T)
        # Defensive: handle both (B, 12, T) and (B, T, 12)
        if out.dim() == 3 and out.shape[1] == 12:
            pred = out
        elif out.dim() == 3 and out.shape[-1] == 12:
            pred = out.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(f"Unexpected ECGrecover output shape: {tuple(out.shape)}")
        if pred.shape[-1] != t:
            pred = F.interpolate(pred.float(), size=t, mode="linear", align_corners=False).to(dtype=lead_i.dtype)
        if return_intermediate:
            return pred, {}
        return pred


class ECGRecoverStyleUNet(nn.Module):
    """ECGrecover-style U-Net completion baseline using only current Lead I."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(1, base_ch)
        self.down1 = nn.Conv1d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv1d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.enc3 = ConvBlock(base_ch * 4, base_ch * 4)
        self.down3 = nn.Conv1d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)
        self.up3 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose1d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        self.out = nn.Conv1d(base_ch, 12, 1)

    def forward(
        self,
        lead_i: torch.Tensor,
        past_12lead: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        patient_meta: Optional[torch.Tensor] = None,
        gt_full: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_intermediate: bool = False,
    ):
        e1 = self.enc1(lead_i)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.up3(b)
        if d3.size(-1) != e3.size(-1):
            d3 = F.interpolate(d3, size=e3.size(-1), mode="linear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        if d2.size(-1) != e2.size(-1):
            d2 = F.interpolate(d2, size=e2.size(-1), mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.size(-1) != e1.size(-1):
            d1 = F.interpolate(d1, size=e1.size(-1), mode="linear", align_corners=False)
        pred = self.out(self.dec1(torch.cat([d1, e1], dim=1)))
        if return_intermediate:
            return pred, {}
        return pred


class ECGNetStyleBiLSTM(nn.Module):
    """ECGNet-style BiLSTM generator baseline for single-lead to 12-lead synthesis."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        hidden = max(32, base_ch * 2)
        self.front = nn.Sequential(
            nn.Conv1d(1, base_ch, 15, padding=7),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch, 7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=base_ch,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 12),
        )

    def forward(
        self,
        lead_i: torch.Tensor,
        past_12lead: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        patient_meta: Optional[torch.Tensor] = None,
        gt_full: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_intermediate: bool = False,
    ):
        h = self.front(lead_i).transpose(1, 2)
        h, _ = self.lstm(h)
        pred = self.out(h).transpose(1, 2)
        if return_intermediate:
            return pred, {}
        return pred


class EKGANStyleGenerator(nn.Module):
    """EKGAN-style dual-generator baseline without adversarial training."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.global_net = ECGRecoverStyleUNet(base_ch=base_ch)
        self.local_net = nn.Sequential(
            ConvBlock(1, base_ch),
            ConvBlock(base_ch, base_ch),
            nn.Conv1d(base_ch, 12, 1),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(1, base_ch, 9, padding=4),
            nn.GELU(),
            nn.Conv1d(base_ch, 12, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        lead_i: torch.Tensor,
        past_12lead: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        patient_meta: Optional[torch.Tensor] = None,
        gt_full: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_intermediate: bool = False,
    ):
        g = self.global_net(lead_i)
        l = self.local_net(lead_i)
        gate = self.gate(lead_i)
        pred = gate * g + (1.0 - gate) * l
        if return_intermediate:
            return pred, {"ekgan_gate": gate}
        return pred


_TF_CONFIGURED = False


def require_tensorflow():
    global _TF_CONFIGURED
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "EKGAN is the official TensorFlow baseline. Install TensorFlow first, "
            "for example: pip install tensorflow"
        ) from e
    if not _TF_CONFIGURED:
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
        _TF_CONFIGURED = True
    return tf


class EKGANLearningRateSchedule:
    """Fixed version of the official EKGAN schedule.

    The released code used a placeholder string for ``path`` and returned a
    one-step, non-cumulative decay. Here ``path`` is the true steps/epoch and
    the 0.95 decay is cumulative after the first 5 epochs.
    """

    def __new__(
        cls,
        initial_learning_rate: float,
        steps_per_epoch: int,
        decay_rate: float = 0.95,
        decay_start_epoch: int = 5,
    ):
        tf = require_tensorflow()

        class _Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self):
                self.initial_learning_rate = float(initial_learning_rate)
                self.steps_per_epoch = float(max(1, steps_per_epoch))
                self.decay_rate = float(decay_rate)
                self.decay_start_epoch = float(decay_start_epoch)

            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                epoch = tf.floor(step / self.steps_per_epoch)
                power = tf.maximum(0.0, epoch - self.decay_start_epoch + 1.0)
                return self.initial_learning_rate * tf.pow(self.decay_rate, power)

        return _Schedule()


class EKGANTFReconAdapter:
    """Official TensorFlow EKGAN generator/discriminator/loss integration."""

    def __init__(self, args: Args, use_prior=False, use_time_delta=False):
        self.tf = require_tensorflow()
        self.use_prior = bool(use_prior)
        self.use_time_delta = bool(use_time_delta)
        # ch0 = current Lead-I image; +ch = previous 12-lead prior; +ch = time-gap.
        self.past_ch = 1 if self.use_prior else None
        self.td_ch = (1 + int(self.use_prior)) if self.use_time_delta else None
        self.in_ch = 1 + int(self.use_prior) + int(self.use_time_delta)
        self.time_len = int(args.official_signal_len)
        self.lambda_ = float(args.ekgan_lambda)
        self.alpha = float(args.ekgan_alpha)
        self.lr = float(args.lr)
        self.decay_rate = float(args.ekgan_lr_decay)
        self.decay_start_epoch = int(args.ekgan_decay_start_epoch)
        self.initializer = self.tf.random_normal_initializer(0.0, 0.02)
        self.loss_object = self.tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.inference_generator = self._load_inference_generator()
        self.label_generator = self._load_label_generator()
        self.discriminator = self._load_discriminator()
        self.inference_generator_optimizer = None
        self.label_generator_optimizer = None
        self.discriminator_optimizer = None

    def _leaky_relu(self):
        try:
            return self.tf.keras.layers.LeakyReLU(negative_slope=0.2)
        except TypeError:
            return self.tf.keras.layers.LeakyReLU(alpha=0.2)

    def _load_inference_generator(self):
        tf = self.tf
        filters_encoder = [64, 128, 256, 512, 1024]
        filters_decoder = [512, 256, 128, 64, 1]
        kernel = (2, 4)
        stride = [2, 2, 2, 2, (1, 2)]

        ig_inputs = tf.keras.Input(shape=(16, self.time_len, self.in_ch), name="ig")
        h = []
        x = tf.keras.layers.Conv2D(
            filters_encoder[0], kernel, stride[0], padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(ig_inputs)
        x = self._leaky_relu()(x)
        h.append(x)
        for i in range(1, 5):
            x = tf.keras.layers.Conv2D(
                filters_encoder[i], kernel, stride[i], padding="same",
                kernel_initializer=self.initializer, use_bias=False
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = self._leaky_relu()(x)
            h.append(x)
        ig_lv = x
        x = tf.keras.layers.Conv2DTranspose(
            filters_decoder[0], kernel, stride[-1], padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        for i in range(1, 4):
            x = tf.keras.layers.Concatenate()([x, h[-i - 1]])
            x = tf.keras.layers.Conv2DTranspose(
                filters_decoder[i], kernel, stride[-i - 1], padding="same",
                kernel_initializer=self.initializer, use_bias=False
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Concatenate()([x, h[0]])
        ig_outputs = tf.keras.layers.Conv2DTranspose(
            filters_decoder[4], kernel, stride[-4], padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(x)
        return tf.keras.Model(inputs=[ig_inputs], outputs=[ig_outputs, ig_lv], name="ekgan_inference_generator")

    def _load_label_generator(self):
        tf = self.tf
        filters_encoder = [64, 128, 256, 512, 1024]
        # Label generator autoencodes its own input, so it must reconstruct all
        # in_ch channels (1 without prior, 2 with the previous-12-lead channel).
        filters_decoder = [512, 256, 128, 64, self.in_ch]
        kernel = (2, 4)
        stride = [2, 2, 2, 2, (1, 2)]

        lg_inputs = tf.keras.Input(shape=(16, self.time_len, self.in_ch), name="lg")
        x = tf.keras.layers.Conv2D(
            filters_encoder[0], kernel, stride[0], padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(lg_inputs)
        x = self._leaky_relu()(x)
        for i in range(1, 5):
            x = tf.keras.layers.Conv2D(
                filters_encoder[i], kernel, stride[i], padding="same",
                kernel_initializer=self.initializer, use_bias=False
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = self._leaky_relu()(x)
        lg_lv = x
        x = tf.keras.layers.Conv2DTranspose(
            filters_decoder[0], kernel, stride[-1], padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        for i in range(1, 4):
            x = tf.keras.layers.Conv2DTranspose(
                filters_decoder[i], kernel, stride[-i - 1], padding="same",
                kernel_initializer=self.initializer, use_bias=False
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
        lg_outputs = tf.keras.layers.Conv2DTranspose(
            filters_decoder[4], kernel, stride[-4], padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(x)
        return tf.keras.Model(inputs=[lg_inputs], outputs=[lg_outputs, lg_lv], name="ekgan_label_generator")

    def _load_discriminator(self):
        tf = self.tf
        filters_encoder = [32, 64, 128, 256, 512]
        filters_decoder = [256, 128, 64, 32, 1]
        kernel = [64, 32, 16, 8, 4]
        stride = [4, 4, 4, 2, 2]

        encoder_inputs = tf.keras.Input(shape=(16, self.time_len, self.in_ch), name="dis")
        target = tf.keras.Input(shape=(16, self.time_len, 1), name="dis_tar")
        x = tf.keras.layers.Concatenate()([encoder_inputs, target])
        h = []
        x = tf.keras.layers.Conv2D(
            filters_encoder[0], (1, kernel[0]), (1, stride[0]), padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(x)
        x = self._leaky_relu()(x)
        h.append(x)
        for i in range(1, 5):
            x = tf.keras.layers.Conv2D(
                filters_encoder[i], (1, kernel[i]), (1, stride[i]), padding="same",
                kernel_initializer=self.initializer, use_bias=False
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = self._leaky_relu()(x)
            h.append(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters_decoder[0], (1, kernel[-1]), (1, stride[-1]), padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        for i in range(1, 4):
            x = tf.keras.layers.Concatenate()([x, h[-i - 1]])
            x = tf.keras.layers.Conv2DTranspose(
                filters_decoder[i], (1, kernel[-i - 1]), (1, stride[-i - 1]), padding="same",
                kernel_initializer=self.initializer, use_bias=False
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Concatenate()([x, h[0]])
        encoder_outputs = tf.keras.layers.Conv2DTranspose(
            filters_decoder[4], (1, kernel[-5]), (1, stride[-4]), padding="same",
            kernel_initializer=self.initializer, use_bias=False
        )(x)
        return tf.keras.Model(inputs=[encoder_inputs, target], outputs=[encoder_outputs], name="ekgan_discriminator")

    def compile_for_training(self, steps_per_epoch: int) -> None:
        tf = self.tf
        lr_schedule = EKGANLearningRateSchedule(
            self.lr,
            steps_per_epoch=steps_per_epoch,
            decay_rate=self.decay_rate,
            decay_start_epoch=self.decay_start_epoch,
        )
        self.inference_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
        self.label_generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)

    def count_params(self) -> int:
        return int(
            self.inference_generator.count_params()
            + self.label_generator.count_params()
            + self.discriminator.count_params()
        )

    def _batch_to_arrays(self, batch: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, int]:
        lead_i = batch["lead_i"].float()
        target = batch["target"].float()
        original_len = int(lead_i.shape[-1])
        if original_len != self.time_len:
            lead_i = F.interpolate(lead_i, size=self.time_len, mode="linear", align_corners=False)
            target = F.interpolate(target, size=self.time_len, mode="linear", align_corners=False)
        lead_np = lead_i[:, 0, :].detach().cpu().numpy().astype(np.float32)
        target_np = target.detach().cpu().numpy().astype(np.float32)
        b = lead_np.shape[0]
        input_image = np.zeros((b, 16, self.time_len, self.in_ch), dtype=np.float32)
        target_image = np.zeros((b, 16, self.time_len, 1), dtype=np.float32)
        input_image[:, 2:14, :, 0] = lead_np[:, None, :]
        if self.use_prior:
            past = batch["past"].float()
            if past.shape[-1] != self.time_len:
                past = F.interpolate(past, size=self.time_len, mode="linear", align_corners=False)
            past_np = past.detach().cpu().numpy().astype(np.float32)
            input_image[:, 2:14, :, self.past_ch] = past_np
        if self.use_time_delta:
            td_np = batch["time_delta"].detach().cpu().numpy().astype(np.float32)  # (B, 1)
            input_image[:, 2:14, :, self.td_ch] = td_np[:, :, None]
        target_image[:, 2:14, :, 0] = target_np
        return input_image, target_image, original_len

    def train_step(self, input_image: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        tf = self.tf
        if self.inference_generator_optimizer is None:
            raise RuntimeError("Call compile_for_training() before EKGAN train_step().")
        input_image_tf = tf.convert_to_tensor(input_image, dtype=tf.float32)
        target_tf = tf.convert_to_tensor(target, dtype=tf.float32)
        with tf.GradientTape() as ig_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as lg_tape:
            ig_output, ig_lv = self.inference_generator(input_image_tf, training=True)
            lg_output, lg_lv = self.label_generator(input_image_tf, training=True)
            disc_real_output = self.discriminator([input_image_tf, target_tf], training=True)
            disc_generated_output = self.discriminator([input_image_tf, ig_output], training=True)

            lg_l1_loss = tf.reduce_mean(tf.abs(input_image_tf - lg_output))
            gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
            ig_l1_loss = tf.reduce_mean(tf.abs(target_tf - ig_output))
            vector_loss = tf.reduce_mean(tf.abs(ig_lv - lg_lv))
            total_ig_loss = ig_l1_loss * self.lambda_ + gan_loss + vector_loss * self.alpha
            real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
            generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
            disc_loss = real_loss + generated_loss

        ig_grads = ig_tape.gradient(total_ig_loss, self.inference_generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        lg_grads = lg_tape.gradient(lg_l1_loss, self.label_generator.trainable_variables)
        self.inference_generator_optimizer.apply_gradients(
            zip(ig_grads, self.inference_generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        self.label_generator_optimizer.apply_gradients(zip(lg_grads, self.label_generator.trainable_variables))
        return {
            "total": float(total_ig_loss.numpy()),
            "gan": float(gan_loss.numpy()),
            "l1": float(ig_l1_loss.numpy()),
            "vector": float(vector_loss.numpy()),
            "disc": float(disc_loss.numpy()),
            "label": float(lg_l1_loss.numpy()),
        }

    def predict_torch_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_image, _, original_len = self._batch_to_arrays(batch)
        pred_tf, _ = self.inference_generator(self.tf.convert_to_tensor(input_image, dtype=self.tf.float32), training=False)
        pred_np = pred_tf.numpy()[:, 2:14, :, 0].astype(np.float32)
        pred = torch.from_numpy(pred_np)
        if pred.shape[-1] != original_len:
            pred = F.interpolate(pred, size=original_len, mode="linear", align_corners=False)
        return pred

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        self.inference_generator.save_weights(str(path / "inference_generator.weights.h5"))
        self.label_generator.save_weights(str(path / "label_generator.weights.h5"))
        self.discriminator.save_weights(str(path / "discriminator.weights.h5"))
        with open(path / "ekgan_config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "time_len": self.time_len,
                    "lambda": self.lambda_,
                    "alpha": self.alpha,
                    "use_prior": self.use_prior,
                    "in_ch": self.in_ch,
                    "repo": EKGAN_REPO,
                },
                f,
                indent=2,
            )
        return path

    def load(self, path: Path) -> None:
        self.inference_generator.load_weights(str(path / "inference_generator.weights.h5"))
        lg_path = path / "label_generator.weights.h5"
        dis_path = path / "discriminator.weights.h5"
        if lg_path.exists():
            self.label_generator.load_weights(str(lg_path))
        if dis_path.exists():
            self.discriminator.load_weights(str(dis_path))


class _MCMADownBlock(nn.Module):
    """Official MCMA downblock: Conv/GELU path + LN/GELU/Conv/GELU path."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 13, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.left = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad),
            nn.GELU(),
        )
        self.right_conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.right_ln = nn.LayerNorm(out_ch)
        self.right_conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad),
            nn.GELU(),
        )
        self.inorm = nn.InstanceNorm1d(out_ch, affine=True, eps=1e-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.left(x)
        right = self.right_conv1(x).transpose(1, 2)
        right = F.gelu(self.right_ln(right)).transpose(1, 2)
        right = self.right_conv2(right)
        return self.inorm(left + right)


class _MCMAUpBlock(nn.Module):
    """Official MCMA upblock with Conv1DTranspose in both branches."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 13, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        out_pad = stride - 1 if stride > 1 else 0
        self.left = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad, output_padding=out_pad),
            nn.GELU(),
        )
        self.right_deconv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=pad, output_padding=out_pad
        )
        self.right_ln = nn.LayerNorm(out_ch)
        self.right_conv = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=pad),
            nn.GELU(),
        )
        self.inorm = nn.InstanceNorm1d(out_ch, affine=True, eps=1e-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.left(x)
        right = self.right_deconv(x).transpose(1, 2)
        right = F.gelu(self.right_ln(right)).transpose(1, 2)
        right = self.right_conv(right)
        return self.inorm(left + right)


def _match_time(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape[-1] == b.shape[-1]:
        return a, b
    t = min(a.shape[-1], b.shape[-1])
    return a[..., :t], b[..., :t]


class MCMAStyleRecon(nn.Module):
    """Faithful PyTorch port of MCMA's official Keras ``modelx`` backbone.

    The official model receives a 12-channel ECG tensor. For the single-lead
    setting, only Lead I is copied into that tensor and the other 11 leads are
    zero padded, matching the MCMA masking/padding recipe.
    """

    def __init__(self, base_ch=None, kernel_size=13, pool_size=2,
                 use_prior=False, use_time_delta=False, meta_dim=0):
        super().__init__()
        self.use_prior = use_prior
        self.use_time_delta = use_time_delta
        self.meta_dim = int(meta_dim)
        in_ch = 12
        if use_prior:
            in_ch += 12             # raw previous 12-lead
        if use_time_delta:
            in_ch += 1              # broadcast normalized time-gap channel
        if self.meta_dim > 0:
            in_ch += self.meta_dim  # broadcast encoded-prior conditioning vector
        filters = [16, 32, 64, 128, 256, 512]
        self.e1 = _MCMADownBlock(in_ch, filters[0], kernel_size, stride=1)
        self.e2 = _MCMADownBlock(filters[0], filters[1], kernel_size, stride=pool_size)
        self.e3 = _MCMADownBlock(filters[1], filters[2], kernel_size, stride=pool_size)
        self.e4 = _MCMADownBlock(filters[2], filters[3], kernel_size, stride=pool_size)
        self.e5 = _MCMADownBlock(filters[3], filters[4], kernel_size, stride=pool_size)
        self.e6 = _MCMADownBlock(filters[4], filters[5], kernel_size, stride=pool_size)

        self.d5_up = _MCMAUpBlock(filters[5], filters[4], kernel_size, stride=pool_size)
        self.d5_skip = _MCMADownBlock(filters[4], filters[4], kernel_size, stride=1)
        self.d4_up = _MCMAUpBlock(filters[4], filters[3], kernel_size, stride=pool_size)
        self.d4_skip = _MCMADownBlock(filters[3], filters[3], kernel_size, stride=1)
        self.d3_up = _MCMAUpBlock(filters[3], filters[2], kernel_size, stride=pool_size)
        self.d3_skip = _MCMADownBlock(filters[2], filters[2], kernel_size, stride=1)
        self.d2_up = _MCMAUpBlock(filters[2], filters[1], kernel_size, stride=pool_size)
        self.d2_skip = _MCMADownBlock(filters[1], filters[1], kernel_size, stride=1)
        self.d1_up = _MCMAUpBlock(filters[1], filters[0], kernel_size, stride=pool_size)
        self.d1_skip = _MCMADownBlock(filters[0], filters[0], kernel_size, stride=1)
        self.out = _MCMADownBlock(filters[0], 12, kernel_size, stride=1)

    def forward(
        self,
        lead_i: torch.Tensor,
        past_12lead: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None,
        patient_meta: Optional[torch.Tensor] = None,
        gt_full: Optional[torch.Tensor] = None,
        teacher_forcing: float = 0.0,
        return_intermediate: bool = False,
        meta: Optional[torch.Tensor] = None,
    ):
        b, _, t = lead_i.shape
        x = torch.zeros(b, 12, t, device=lead_i.device, dtype=lead_i.dtype)
        x[:, IDX["I"] : IDX["I"] + 1] = lead_i

        if self.use_prior:
            if past_12lead is None:
                past_12lead = torch.zeros(b, 12, t, device=lead_i.device, dtype=lead_i.dtype)
            elif past_12lead.shape[-1] != t:
                past_12lead = F.interpolate(
                    past_12lead.float(), size=t, mode="linear", align_corners=False
                ).to(dtype=lead_i.dtype)
            x = torch.cat([x, past_12lead], dim=1)

        if self.use_time_delta:
            if time_delta is None:
                time_delta = torch.zeros(b, 1, device=lead_i.device, dtype=lead_i.dtype)
            td_ch = time_delta.view(b, 1, 1).expand(b, 1, t).to(dtype=x.dtype)
            x = torch.cat([x, td_ch], dim=1)

        if self.meta_dim > 0:
            if meta is None:
                meta = torch.zeros(b, self.meta_dim, device=lead_i.device, dtype=lead_i.dtype)
            cond_ch = meta.view(b, self.meta_dim, 1).expand(b, self.meta_dim, t).to(dtype=x.dtype)
            x = torch.cat([x, cond_ch], dim=1)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        u, s = _match_time(self.d5_up(e6), self.d5_skip(e5))
        d5 = u + s
        u, s = _match_time(self.d4_up(d5), self.d4_skip(e4))
        d4 = u + s
        u, s = _match_time(self.d3_up(d4), self.d3_skip(e3))
        d3 = u + s
        u, s = _match_time(self.d2_up(d3), self.d2_skip(e2))
        d2 = u + s
        u, s = _match_time(self.d1_up(d2), self.d1_skip(e1))
        d1 = u + s

        pred = self.out(d1)
        if pred.shape[-1] != t:
            pred = F.interpolate(pred, size=t, mode="linear", align_corners=False)
        if return_intermediate:
            return pred, {}
        return pred


class ECGClassifier(nn.Module):
    def __init__(self, n_labels: int, base_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, base_ch, 15, padding=7),
            nn.BatchNorm1d(base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch * 2, 9, stride=2, padding=4),
            nn.BatchNorm1d(base_ch * 2),
            nn.GELU(),
            nn.Conv1d(base_ch * 2, base_ch * 4, 9, stride=2, padding=4),
            nn.BatchNorm1d(base_ch * 4),
            nn.GELU(),
            nn.Conv1d(base_ch * 4, base_ch * 4, 7, stride=2, padding=3),
            nn.BatchNorm1d(base_ch * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 4, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def instantiate_base(cls, args: Args, meta_dim: int) -> nn.Module:
    attempts = [
        dict(
            base_ch=args.base_ch,
            res_base_ch=args.res_base_ch,
            residual_scale=args.residual_scale,
            meta_dim=meta_dim,
        ),
        dict(base_ch=args.base_ch, meta_dim=meta_dim),
        dict(meta_dim=meta_dim),
        {},
    ]
    last_err = None
    for kw in attempts:
        try:
            return cls(**kw)
        except TypeError as e:
            last_err = e
    raise TypeError(f"Could not instantiate {cls.__name__}: {last_err}")


TF_RECON_METHODS = {"ekgan", "ekgan_prior"}


def is_tf_recon_method(name: str) -> bool:
    return name in TF_RECON_METHODS


def make_tf_model(name: str, args: Args) -> EKGANTFReconAdapter:
    if name == "ekgan":
        return EKGANTFReconAdapter(args, use_prior=False, use_time_delta=False)
    if name == "ekgan_prior":
        return EKGANTFReconAdapter(args, use_prior=True, use_time_delta=True)
    raise ValueError(f"Unknown TensorFlow method: {name}")


def make_model(name: str, args: Args) -> nn.Module:
    if name == "lead_i":
        return LeadIOnlyRecon(instantiate_base(CascadeECGRecon, args, meta_dim=0), use_meta=False)
    if name == "ecgrecover":
        return ECGRecoverStyleUNet(base_ch=args.base_ch)
    if name == "ecgrecover_official":
        return OfficialECGRecover(signal_len=args.official_signal_len)
    if name == "ecgnet":
        return ECGNetStyleBiLSTM(base_ch=args.base_ch)
    if name == "ekgan_torch":
        return EKGANStyleGenerator(base_ch=args.base_ch)
    if name == "mcma":
        return MCMAStyleRecon(base_ch=args.base_ch, use_prior=False)
    if name == "mcma_prior":
        return MCMAStyleRecon(base_ch=args.base_ch, use_prior=True, use_time_delta=True)
    if name == "mcma_prior_enc":
        base = MCMAStyleRecon(base_ch=args.base_ch, meta_dim=args.prior_dim)
        return PriorConditionedRecon(base, prior_dim=args.prior_dim, use_meta=False)
    if name == "lead_i_meta":
        return LeadIOnlyRecon(instantiate_base(CascadeECGRecon, args, meta_dim=6), use_meta=True)
    if name == "prime_ecg":
        base = instantiate_base(CascadeECGRecon, args, meta_dim=args.prior_dim)
        return PriorConditionedRecon(base, prior_dim=args.prior_dim, use_meta=False)
    if name == "prime_ecg_meta":
        base = instantiate_base(CascadeECGRecon, args, meta_dim=args.prior_dim + 6)
        return PriorConditionedRecon(base, prior_dim=args.prior_dim, use_meta=True)
    if name == "no_physics":
        base = instantiate_base(VanillaECGRecon, args, meta_dim=args.prior_dim + 6)
        return PriorConditionedRecon(base, prior_dim=args.prior_dim, use_meta=True)
    if name == "concat":
        return ConcatUNetRecon(base_ch=args.base_ch)
    raise ValueError(f"Unknown method: {name}")


def model_uses_meta(name: str) -> bool:
    return name in {"lead_i_meta", "prime_ecg_meta", "no_physics"}


def model_uses_prior(name: str) -> bool:
    # ekgan_prior is a TF model that reads ``past`` internally in
    # _batch_to_arrays; listing it here is harmless and documents intent.
    return name in {"prime_ecg", "prime_ecg_meta", "no_physics", "concat",
                    "mcma_prior", "ekgan_prior", "mcma_prior_enc"}


def method_info(name: str) -> Dict[str, str]:
    info = {
        "lead_i": ("internal", "Cascade Lead-I baseline", "current Lead I"),
        "ecgrecover": ("prior_baseline", "ECGrecover-style U-Net completion (re-impl)", "current Lead I"),
        "ecgrecover_official": ("prior_baseline", "ECGrecover OFFICIAL (UMMISCO/ecgrecover)", "current Lead I"),
        "ecgnet": ("prior_baseline", "ECGNet-style BiLSTM generator", "current Lead I"),
        "ekgan": ("prior_baseline", "EKGAN OFFICIAL TensorFlow GAN", "current Lead I tiled to 12 padded rows"),
        "ekgan_torch": ("prior_baseline", "EKGAN-style PyTorch generator (non-adversarial re-impl)", "current Lead I"),
        "mcma": ("prior_baseline", "MCMA-style masked multi-channel autoencoder", "current Lead I"),
        "mcma_prior": ("fair_info_baseline", "MCMA + prev-12-lead + time_delta (raw channel stack)", "current Lead I + previous 12-lead + time_delta"),
        "mcma_prior_enc": ("fair_info_baseline", "MCMA + PRIME-style encoded prior + time_delta", "current Lead I + previous 12-lead + time_delta"),
        "ekgan_prior": ("fair_info_baseline", "EKGAN OFFICIAL + prev-12-lead + time_delta (channel stack)", "current Lead I + previous 12-lead + time_delta"),
        "lead_i_meta": ("internal", "Cascade Lead-I + metadata", "current Lead I + metadata"),
        "concat": ("fair_info_baseline", "Channel-wise concat baseline", "current Lead I + previous 12-lead"),
        "prime_ecg": ("ours", "PRIME-ECG without metadata", "current Lead I + previous 12-lead"),
        "prime_ecg_meta": ("ours", "PRIME-ECG full", "current Lead I + previous 12-lead + metadata"),
        "no_physics": ("ablation", "No-physics decoder ablation", "current Lead I + previous 12-lead + metadata"),
        "past_as_output": ("sanity", "Copy previous 12-lead", "previous 12-lead"),
    }
    family, description, input_info = info.get(name, ("unknown", name, "unknown"))
    return {"baseline_family": family, "description": description, "input_info": input_info}


def environment_info(args: Args) -> Dict[str, str]:
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    tf_status = "not_installed"
    if importlib.util.find_spec("tensorflow") is not None:
        try:
            import tensorflow as _tf_env  # type: ignore
            tf_status = getattr(_tf_env, "__version__", "installed")
        except Exception as e:  # noqa: BLE001
            tf_status = f"import_failed:{e}"
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "tensorflow": tf_status,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device": cuda_name,
        "ecgrecover_repo": ECGRECOVER_REPO,
        "ekgan_repo": EKGAN_REPO,
        "wfdb": getattr(wfdb, "__version__", "unknown"),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "epochs": str(args.epochs),
        "batch_size": str(args.batch_size),
        "lr": str(args.lr),
        "weight_decay": str(args.weight_decay),
        "amp": str(args.amp),
        "seed": str(args.seed),
    }


def unwrap_pred(out):
    if isinstance(out, tuple):
        return out[0], out[1] if len(out) > 1 else None
    return out, None


def fallback_loss(pred: torch.Tensor, target: torch.Tensor, lead_i: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target) + 0.2 * F.l1_loss(
        pred[:, IDX["I"] : IDX["I"] + 1], lead_i
    )


def ecgrecover_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Loss used in the ECGrecover paper: MSE - 0.1 * Pearson correlation.

    Applied per-(batch, lead), then averaged. Matches UMMISCO/ecgrecover's
    ``RMSE_Loss`` in ``tools/LossFunction.py``.

    Shapes: pred, target are (B, 12, T).
    """
    mse = torch.nan_to_num(torch.mean((target - pred) ** 2, dim=2))
    t_mean = target.mean(dim=2)
    p_mean = pred.mean(dim=2)
    t_std = torch.sqrt(torch.var(target, dim=2))
    p_std = torch.sqrt(torch.var(pred, dim=2))
    num = torch.sum((target - t_mean.unsqueeze(2)) * (pred - p_mean.unsqueeze(2)), dim=2)
    corr = torch.nan_to_num(num / (target.shape[-1] * t_std * p_std))
    loss = mse - 0.1 * corr
    return torch.mean(loss)


def mcma_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Official MCMA training objective: mean squared reconstruction error."""
    return F.mse_loss(pred, target)


def train_recon(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: Args,
    out_dir: Path,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # LSTM-based ECGNet baseline can crash with cuDNN under fp16 — disable AMP for it.
    use_amp = bool(args.amp and device.type == "cuda" and name != "ecgnet")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    lead_w = get_lead_weight_tensor(device=device)
    best_path = out_dir / f"{name}_best.pt"
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        if args.teacher_forcing:
            tf = max(0.0, 1.0 - (ep - 1) / max(1, args.epochs * args.tf_decay_ratio))
        else:
            tf = 0.0
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{name} ep{ep} train", leave=False):
            x = batch["lead_i"].to(device)
            y = batch["target"].to(device)
            past = batch["past"].to(device) if model_uses_prior(name) else None
            td = batch["time_delta"].to(device) if model_uses_prior(name) else None
            meta = batch["meta"].to(device) if model_uses_meta(name) else None
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(
                    x,
                    past_12lead=past,
                    time_delta=td,
                    patient_meta=meta,
                    gt_full=y,
                    teacher_forcing=tf,
                    return_intermediate=True,
                )
                pred, inter = unwrap_pred(out)
                if name == "ecgrecover_official":
                    # use the ECGrecover paper's own loss (MSE - 0.1*PCC)
                    loss = ecgrecover_loss(pred, y)
                elif name == "mcma":
                    loss = mcma_loss(pred, y)
                else:
                    try:
                        loss = cascade_loss(pred, y, intermediate=inter, lead_w=lead_w)
                    except Exception:
                        loss = fallback_loss(pred, y, x)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_recon_model(name, model, val_loader, args)
        val_score = val_metrics["mae_all"]
        print(
            f"[{name}] ep={ep:03d} train={np.mean(losses):.4f} "
            f"val_mae={val_metrics['mae_all']:.4f} val_r={val_metrics['pearson_all']:.4f}"
        )
        torch.save(model.state_dict(), out_dir / f"{name}_last.pt")
        if val_score < best:
            best = val_score
            torch.save(model.state_dict(), best_path)
    return best_path


def train_tf_recon(
    name: str,
    model: EKGANTFReconAdapter,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: Args,
    out_dir: Path,
) -> Path:
    model.compile_for_training(steps_per_epoch=max(1, len(train_loader)))
    best_path = out_dir / f"{name}_best_tf"
    last_path = out_dir / f"{name}_last_tf"
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        losses = []
        gan_losses = []
        vector_losses = []
        for batch in tqdm(train_loader, desc=f"{name} ep{ep} train", leave=False):
            input_image, target, _ = model._batch_to_arrays(batch)
            logs = model.train_step(input_image, target)
            if math.isfinite(logs["total"]):
                losses.append(logs["total"])
                gan_losses.append(logs["gan"])
                vector_losses.append(logs["vector"])

        val_metrics = evaluate_tf_recon_model(name, model, val_loader, args)
        val_score = val_metrics["mae_all"]
        train_mean = float(np.mean(losses)) if losses else float("nan")
        gan_mean = float(np.mean(gan_losses)) if gan_losses else float("nan")
        vec_mean = float(np.mean(vector_losses)) if vector_losses else float("nan")
        print(
            f"[{name}] ep={ep:03d} train={train_mean:.4f} gan={gan_mean:.4f} "
            f"vec={vec_mean:.4f} val_mae={val_metrics['mae_all']:.4f} "
            f"val_r={val_metrics['pearson_all']:.4f}"
        )
        model.save(last_path)
        if val_score < best:
            best = val_score
            model.save(best_path)
    return best_path


@torch.no_grad()
def evaluate_tf_recon_model(
    name: str,
    model: EKGANTFReconAdapter,
    loader: DataLoader,
    args: Args,
) -> Dict[str, float]:
    stats = ReconStats()
    for batch in tqdm(loader, desc=f"eval {name}", leave=False):
        pred = model.predict_torch_batch(batch)
        stats.update(pred, batch["target"])
    return stats.compute()


def evaluate_any_recon_model(
    name: str,
    model: Any,
    loader: DataLoader,
    args: Args,
) -> Dict[str, float]:
    if is_tf_recon_method(name):
        return evaluate_tf_recon_model(name, model, loader, args)
    return evaluate_recon_model(name, model, loader, args)


@torch.no_grad()
def evaluate_recon_model(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    args: Args,
    force_zero_time: bool = False,
    force_zero_meta: bool = False,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    stats = ReconStats()
    for batch in tqdm(loader, desc=f"eval {name}", leave=False):
        x = batch["lead_i"].to(device)
        y = batch["target"].to(device)
        past = batch["past"].to(device) if model_uses_prior(name) else None
        td = batch["time_delta"].to(device) if model_uses_prior(name) else None
        meta = batch["meta"].to(device) if model_uses_meta(name) else None
        if force_zero_time and td is not None:
            td = torch.zeros_like(td)
        if force_zero_meta and meta is not None:
            meta = torch.zeros_like(meta)
        pred, _ = unwrap_pred(model(x, past_12lead=past, time_delta=td, patient_meta=meta))
        stats.update(pred, y)
    return stats.compute()


@torch.no_grad()
def evaluate_past_as_output(loader: DataLoader) -> Dict[str, float]:
    stats = ReconStats()
    for batch in tqdm(loader, desc="eval past_as_output", leave=False):
        stats.update(batch["past"], batch["target"])
    return stats.compute()


class ReconStats:
    def __init__(self):
        self.n = 0
        self.abs_sum = torch.zeros(12)
        self.sq_sum = torch.zeros(12)
        self.corr_sum = torch.zeros(12)
        self.corr_n = torch.zeros(12)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().cpu().float()
        target = target.detach().cpu().float()
        b = pred.size(0)
        diff = pred - target
        self.abs_sum += diff.abs().mean(dim=2).sum(dim=0)
        self.sq_sum += diff.pow(2).mean(dim=2).sum(dim=0)
        for bi in range(b):
            for li in range(12):
                p = pred[bi, li].numpy()
                y = target[bi, li].numpy()
                if p.std() > 1e-8 and y.std() > 1e-8:
                    c = np.corrcoef(p, y)[0, 1]
                    if np.isfinite(c):
                        self.corr_sum[li] += float(c)
                        self.corr_n[li] += 1
        self.n += b

    def compute(self) -> Dict[str, float]:
        mae = self.abs_sum / max(1, self.n)
        mse = self.sq_sum / max(1, self.n)
        rmse = torch.sqrt(mse)
        corr = self.corr_sum / torch.clamp(self.corr_n, min=1)

        def idxs(names: Sequence[str]) -> List[int]:
            return [IDX[x] for x in names]

        out = {
            "mae_all": float(mae.mean()),
            "mse_all": float(mse.mean()),
            "rmse_all": float(rmse.mean()),
            "pearson_all": float(corr.mean()),
            # MCMA-style aggregate aliases ----------------------------------
            "mse_mean": float(mse.mean()),   # MCMA Table "Mean" column (MSE row)
            "pcc_mean": float(corr.mean()),  # MCMA Table "Mean" column (PCC row)
            # ---------------------------------------------------------------
            "mae_limb": float(mae[idxs(LIMB_LEADS)].mean()),
            "mae_v1_v4": float(mae[idxs(ANTERIOR_LEADS)].mean()),
            "mae_chest": float(mae[idxs(CHEST_LEADS)].mean()),
            "pearson_v1_v4": float(corr[idxs(ANTERIOR_LEADS)].mean()),
            "pearson_chest": float(corr[idxs(CHEST_LEADS)].mean()),
        }
        for i, lead in enumerate(LEADS):
            out[f"mae_{lead}"] = float(mae[i])
            out[f"mse_{lead}"] = float(mse[i])
            out[f"rmse_{lead}"] = float(rmse[i])
            out[f"pearson_{lead}"] = float(corr[i])
            # MCMA-style per-lead aliases (MSE / PCC) -----------------------
            out[f"pcc_{lead}"] = float(corr[i])
        return out

    @staticmethod
    def format_mcma_table(rows: List[Dict[str, float]]) -> str:
        """Format a list of metric dicts as an MCMA-style LaTeX-friendly table.

        Each row should contain a 'method' key plus mse_{lead} / pcc_{lead}.
        Output columns: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6, Mean.
        Two blocks are produced: one for MSE, one for PCC.
        """
        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF",
                      "V1", "V2", "V3", "V4", "V5", "V6"]
        header = ["Method"] + lead_order + ["Mean"]

        def fmt_block(metric_prefix: str, mean_key: str) -> str:
            lines = ["\t".join(header)]
            for r in rows:
                cells = [str(r.get("method", "?"))]
                for ld in lead_order:
                    v = r.get(f"{metric_prefix}_{ld}")
                    cells.append(f"{v:.4f}" if v is not None else "-")
                m = r.get(mean_key)
                cells.append(f"{m:.4f}" if m is not None else "-")
                lines.append("\t".join(cells))
            return "\n".join(lines)

        return ("[MSE per lead]\n" + fmt_block("mse", "mse_mean")
                + "\n\n[PCC per lead]\n" + fmt_block("pcc", "pcc_mean"))


def make_loader(
    data_dir: Path,
    pairs: pd.DataFrame,
    pool_pairs: pd.DataFrame,
    labels: Sequence[str],
    meta_stats: Dict[str, float],
    args: Args,
    past_mode: str = "self",
    train: bool = False,
    zero_time: bool = False,
    zero_meta: bool = False,
) -> DataLoader:
    ds = PriorPairDataset(
        data_dir,
        pairs,
        label_names=labels,
        meta_stats=meta_stats,
        pool_pairs=pool_pairs,
        past_mode=past_mode,
        zero_time=zero_time,
        zero_meta=zero_meta,
        seed=args.seed + (11 if train else 101),
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def gap_name(delta_days: float) -> str:
    for name, lo, hi in GAP_BINS:
        if delta_days > lo and delta_days <= hi:
            return name
        if name == "0-1d" and delta_days >= lo and delta_days <= hi:
            return name
    return "unknown"


@torch.no_grad()
def temporal_gap_eval(
    name: str,
    model: Optional[nn.Module],
    pairs: pd.DataFrame,
    data_dir: Path,
    labels: Sequence[str],
    meta_stats: Dict[str, float],
    args: Args,
    is_past_copy: bool = False,
) -> List[Dict[str, float]]:
    rows = []
    for bin_name, lo, hi in GAP_BINS:
        sub = pairs[(pairs["delta_days"] > lo) & (pairs["delta_days"] <= hi)]
        if bin_name == "0-1d":
            sub = pairs[(pairs["delta_days"] >= lo) & (pairs["delta_days"] <= hi)]
        if len(sub) == 0:
            continue
        loader = make_loader(data_dir, sub, pairs, labels, meta_stats, args, train=False)
        metrics = evaluate_past_as_output(loader) if is_past_copy else evaluate_any_recon_model(name, model, loader, args)
        metrics.update({"method": name, "gap_bin": bin_name, "n_pairs": len(sub)})
        rows.append(metrics)
    return rows


def train_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_labels: int,
    args: Args,
    out_dir: Path,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = ECGClassifier(n_labels=n_labels, base_ch=args.base_ch).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_path = out_dir / "classifier_best.pt"
    best = float("inf")
    for ep in range(1, args.classifier_epochs + 1):
        clf.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"clf ep{ep} train", leave=False):
            x = batch["target"].to(device)
            y = batch["labels"].to(device)
            if args.classifier_lead_dropout > 0:
                keep = (torch.rand(x.size(0), 12, 1, device=device) > args.classifier_lead_dropout).float()
                keep[:, IDX["I"], :] = 1.0
                x = x * keep
            loss = F.binary_cross_entropy_with_logits(clf(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_loss = classifier_loss(clf, val_loader)
        print(f"[classifier] ep={ep:03d} train={np.mean(losses):.4f} val={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(clf.state_dict(), best_path)
    return best_path


@torch.no_grad()
def classifier_loss(clf: nn.Module, loader: DataLoader) -> float:
    device = next(clf.parameters()).device
    clf.eval()
    losses = []
    for batch in loader:
        x = batch["target"].to(device)
        y = batch["labels"].to(device)
        losses.append(float(F.binary_cross_entropy_with_logits(clf(x), y).detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def metric_multilabel(y_true: np.ndarray, y_prob: np.ndarray, labels: Sequence[str]) -> Dict[str, float]:
    out = {}
    aurocs, auprcs, f1s = [], [], []
    y_pred = (y_prob >= 0.5).astype(np.float32)
    for i, label in enumerate(labels):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) < 2:
            auc = aupr = f1 = float("nan")
        else:
            auc = float(roc_auc_score(yt, yp)) if roc_auc_score else float("nan")
            aupr = float(average_precision_score(yt, yp)) if average_precision_score else float("nan")
            f1 = float(f1_score(yt, y_pred[:, i])) if f1_score else float("nan")
        out[f"auroc_{label}"] = auc
        out[f"auprc_{label}"] = aupr
        out[f"f1_{label}"] = f1
        aurocs.append(auc)
        auprcs.append(aupr)
        f1s.append(f1)
    out["macro_auroc"] = float(np.nanmean(aurocs))
    out["macro_auprc"] = float(np.nanmean(auprcs))
    out["macro_f1"] = float(np.nanmean(f1s))
    return out


@torch.no_grad()
def predict_recon_batch(name: str, model: Any, batch: Dict[str, torch.Tensor], args: Args) -> torch.Tensor:
    if is_tf_recon_method(name):
        return model.predict_torch_batch(batch)
    device = next(model.parameters()).device
    x = batch["lead_i"].to(device)
    past = batch["past"].to(device) if model_uses_prior(name) else None
    td = batch["time_delta"].to(device) if model_uses_prior(name) else None
    meta = batch["meta"].to(device) if model_uses_meta(name) else None
    pred, _ = unwrap_pred(model(x, past_12lead=past, time_delta=td, patient_meta=meta))
    return pred


def mask_from_leads(leads: Sequence[str], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, 12, 1, device=device)
    for lead in leads:
        mask[:, IDX[lead], :] = 1.0
    return mask


@torch.no_grad()
def eval_classifier_source(
    source_name: str,
    clf: nn.Module,
    loader: DataLoader,
    labels: Sequence[str],
    recon_model: Optional[nn.Module] = None,
    recon_name: str = "prime_ecg_meta",
    selected: Optional[Dict[str, List[str]]] = None,
    args: Optional[Args] = None,
) -> Dict[str, float]:
    device = next(clf.parameters()).device
    clf.eval()
    if recon_model is not None:
        recon_model.to(device).eval()
    y_true_all, y_prob_all = [], []
    for batch in tqdm(loader, desc=f"classify {source_name}", leave=False):
        y_true = batch["labels"].numpy()
        if source_name == "real12":
            x = batch["target"].to(device)
            prob = torch.sigmoid(clf(x)).cpu().numpy()
        elif source_name == "lead_i":
            x = torch.zeros_like(batch["target"]).to(device)
            x[:, IDX["I"] : IDX["I"] + 1] = batch["lead_i"].to(device)
            prob = torch.sigmoid(clf(x)).cpu().numpy()
        elif source_name == "past12":
            x = batch["past"].to(device)
            prob = torch.sigmoid(clf(x)).cpu().numpy()
        else:
            assert recon_model is not None and args is not None
            recon = predict_recon_batch(recon_name, recon_model, batch, args).to(device)
            if selected is None:
                prob = torch.sigmoid(clf(recon)).cpu().numpy()
            else:
                per_label = []
                for label in labels:
                    mask = mask_from_leads(selected[label], device)
                    p = torch.sigmoid(clf(recon * mask))[:, labels.index(label)]
                    per_label.append(p.cpu().numpy())
                prob = np.stack(per_label, axis=1)
        y_true_all.append(y_true)
        y_prob_all.append(prob)
    y_true_np = np.concatenate(y_true_all, axis=0)
    y_prob_np = np.concatenate(y_prob_all, axis=0)
    out = metric_multilabel(y_true_np, y_prob_np, labels)
    out["source"] = source_name
    return out


@torch.no_grad()
def select_leads_by_validation(
    clf: nn.Module,
    recon_model: nn.Module,
    recon_name: str,
    val_loader: DataLoader,
    labels: Sequence[str],
    args: Args,
) -> Dict[str, List[str]]:
    selected = {}
    device = next(clf.parameters()).device
    for label in labels:
        scores = []
        for lead in LEADS:
            probs, ys = [], []
            for batch in val_loader:
                recon = predict_recon_batch(recon_name, recon_model, batch, args).to(device)
                mask = mask_from_leads([lead], device)
                p = torch.sigmoid(clf(recon * mask))[:, labels.index(label)].cpu().numpy()
                probs.append(p)
                ys.append(batch["labels"][:, labels.index(label)].numpy())
            y = np.concatenate(ys)
            p = np.concatenate(probs)
            if len(np.unique(y)) < 2 or roc_auc_score is None:
                score = 0.0
            else:
                score = float(roc_auc_score(y, p))
            scores.append((score, lead))
        scores.sort(reverse=True)
        keep = [lead for _, lead in scores[: args.select_top_k]]
        if "I" not in keep:
            keep[-1] = "I"
        selected[label] = sorted(set(keep), key=LEADS.index)
    return selected


def random_same_size_selection(selected: Dict[str, List[str]], seed: int) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    out = {}
    for label, leads in selected.items():
        k = len(leads)
        sample = rng.sample(LEADS, k)
        if "I" not in sample:
            sample[-1] = "I"
        out[label] = sorted(set(sample), key=LEADS.index)
    return out


def save_csv(rows: List[Dict[str, float]], path: Path) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[SAVE] {path}")


def attach_method_metadata(row: Dict[str, float], method: str, params: int = 0, train_seconds: float = 0.0, ckpt: str = "") -> Dict[str, float]:
    row.update(method_info(method))
    row["params"] = params
    row["params_m"] = params / 1e6
    row["train_seconds"] = train_seconds
    row["best_checkpoint"] = ckpt
    return row


def run() -> None:
    args = parse_args()
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(asdict(args), f, indent=2)
    env = environment_info(args)
    with open(out_dir / "experiment_environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    pd.DataFrame([env]).to_csv(out_dir / "experiment_environment.csv", index=False)

    labels = [x.strip() for x in args.disease_labels.split(",") if x.strip()]
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    df = add_labels(df, data_dir, labels)
    fname_col = "filename_hr" if args.use_hr else "filename_lr"

    train_pairs = build_pairs(df, fname_col, parse_int_list(args.train_folds), labels)
    val_pairs = build_pairs(df, fname_col, parse_int_list(args.val_folds), labels)
    test_pairs = build_pairs(df, fname_col, parse_int_list(args.test_folds), labels)
    train_pairs = limit_df(train_pairs, args.max_train_pairs)
    val_pairs = limit_df(val_pairs, args.max_val_pairs)
    test_pairs = limit_df(test_pairs, args.max_test_pairs)
    assert_no_patient_overlap(train_pairs, val_pairs, test_pairs)

    print(f"[DATA] train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    train_pairs.to_csv(out_dir / "train_pairs.csv", index=False)
    val_pairs.to_csv(out_dir / "val_pairs.csv", index=False)
    test_pairs.to_csv(out_dir / "test_pairs.csv", index=False)

    meta_stats = compute_meta_stats(train_pairs)
    with open(out_dir / "meta_stats.json", "w", encoding="utf-8") as f:
        json.dump(meta_stats, f, indent=2)

    train_loader = make_loader(data_dir, train_pairs, train_pairs, labels, meta_stats, args, train=True)
    val_loader = make_loader(data_dir, val_pairs, train_pairs, labels, meta_stats, args)
    test_loader = make_loader(data_dir, test_pairs, train_pairs, labels, meta_stats, args)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    recon_rows = []
    method_rows = []
    trained_paths = {}

    for method in methods:
        if is_tf_recon_method(method):
            model = make_tf_model(method, args)
            n_params = model.count_params()
        else:
            model = make_model(method, args)
            n_params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] {method} params={n_params / 1e6:.3f}M")
        t0 = time.time()
        if is_tf_recon_method(method):
            path = train_tf_recon(method, model, train_loader, val_loader, args, out_dir)
        else:
            path = train_recon(method, model, train_loader, val_loader, args, out_dir)
        elapsed = time.time() - t0
        trained_paths[method] = str(path)
        if is_tf_recon_method(method):
            model.load(Path(path))
        else:
            model.load_state_dict(torch.load(path, map_location="cpu"))
        metrics = evaluate_any_recon_model(method, model, test_loader, args)
        metrics.update({"method": method, "past_mode": "self", "ablation": "none"})
        attach_method_metadata(metrics, method, n_params, elapsed, str(path))
        recon_rows.append(metrics)
        method_row = {
            "method": method,
            "params": n_params,
            "params_m": n_params / 1e6,
            "train_seconds": elapsed,
            "best_checkpoint": str(path),
        }
        method_row.update(method_info(method))
        method_rows.append(method_row)

    past_metrics = evaluate_past_as_output(test_loader)
    past_metrics.update({"method": "past_as_output", "past_mode": "self", "ablation": "none"})
    attach_method_metadata(past_metrics, "past_as_output", 0, 0.0, "")
    recon_rows.append(past_metrics)
    method_row = {"method": "past_as_output", "params": 0, "params_m": 0.0, "train_seconds": 0.0, "best_checkpoint": ""}
    method_row.update(method_info("past_as_output"))
    method_rows.append(method_row)
    save_csv(method_rows, out_dir / "method_summary.csv")

    prime_ecg_name = "prime_ecg_meta" if "prime_ecg_meta" in trained_paths else ("prime_ecg" if "prime_ecg" in trained_paths else None)
    prime_ecg_model = None
    if prime_ecg_name:
        prime_ecg_model = make_model(prime_ecg_name, args)
        prime_ecg_model.load_state_dict(torch.load(trained_paths[prime_ecg_name], map_location="cpu"))

        for mode in ["self", "random", "demo", "zero"]:
            loader = make_loader(data_dir, test_pairs, train_pairs, labels, meta_stats, args, past_mode=mode)
            metrics = evaluate_recon_model(prime_ecg_name, prime_ecg_model, loader, args)
            metrics.update({"method": prime_ecg_name, "past_mode": mode, "ablation": f"{mode}_prior"})
            attach_method_metadata(metrics, prime_ecg_name)
            recon_rows.append(metrics)

        loader_zero_time = make_loader(data_dir, test_pairs, train_pairs, labels, meta_stats, args, zero_time=True)
        metrics = evaluate_recon_model(prime_ecg_name, prime_ecg_model, loader_zero_time, args)
        metrics.update({"method": prime_ecg_name, "past_mode": "self", "ablation": "zero_time_delta"})
        attach_method_metadata(metrics, prime_ecg_name)
        recon_rows.append(metrics)

        if model_uses_meta(prime_ecg_name):
            loader_zero_meta = make_loader(data_dir, test_pairs, train_pairs, labels, meta_stats, args, zero_meta=True)
            metrics = evaluate_recon_model(prime_ecg_name, prime_ecg_model, loader_zero_meta, args)
            metrics.update({"method": prime_ecg_name, "past_mode": "self", "ablation": "zero_metadata"})
            attach_method_metadata(metrics, prime_ecg_name)
            recon_rows.append(metrics)

    save_csv(recon_rows, out_dir / "reconstruction_results.csv")

    # ------------------------------------------------------------------
    # MCMA-style per-lead table (MSE + PCC for I..V6 + Mean)
    # Filter to main reconstruction comparison rows (skip ablations/zero_*).
    # ------------------------------------------------------------------
    try:
        main_rows = [
            r for r in recon_rows
            if r.get("past_mode", "self") == "self"
            and r.get("ablation") in (None, "", "none")
        ]
        if main_rows:
            mcma_table = ReconStats.format_mcma_table(main_rows)
            (out_dir / "reconstruction_table_mcma_style.txt").write_text(
                mcma_table, encoding="utf-8"
            )
            print("\n" + mcma_table + "\n")
    except Exception as e:
        print(f"[warn] could not write MCMA-style table: {e}")

    temporal_rows = []
    temporal_rows.extend(temporal_gap_eval("past_as_output", None, test_pairs, data_dir, labels, meta_stats, args, True))
    if prime_ecg_model is not None and prime_ecg_name is not None:
        temporal_rows.extend(temporal_gap_eval(prime_ecg_name, prime_ecg_model, test_pairs, data_dir, labels, meta_stats, args))
    save_csv(temporal_rows, out_dir / "temporal_gap_results.csv")

    with open(out_dir / "trained_paths.json", "w", encoding="utf-8") as f:
        json.dump(trained_paths, f, indent=2)

    if args.run_classification:
        clf_path = train_classifier(train_loader, val_loader, len(labels), args, out_dir)
        clf = ECGClassifier(n_labels=len(labels), base_ch=args.base_ch)
        clf.load_state_dict(torch.load(clf_path, map_location="cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.to(device)
        class_rows = []
        for source in ["real12", "lead_i", "past12"]:
            class_rows.append(eval_classifier_source(source, clf, test_loader, labels))

        for method, ckpt_path in trained_paths.items():
            if is_tf_recon_method(method):
                recon_model = make_tf_model(method, args)
                recon_model.load(Path(ckpt_path))
            else:
                recon_model = make_model(method, args)
                recon_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                recon_model.to(device)
            row = eval_classifier_source(
                f"recon_{method}",
                clf,
                test_loader,
                labels,
                recon_model,
                method,
                None,
                args,
            )
            row["method"] = method
            row.update(method_info(method))
            class_rows.append(row)

        if prime_ecg_model is not None and prime_ecg_name is not None:
            prime_ecg_model.to(device)
            selected = select_leads_by_validation(clf, prime_ecg_model, prime_ecg_name, val_loader, labels, args)
            random_selected = random_same_size_selection(selected, args.seed)
            with open(out_dir / "selected_leads.json", "w", encoding="utf-8") as f:
                json.dump(selected, f, indent=2)
            with open(out_dir / "random_selected_leads.json", "w", encoding="utf-8") as f:
                json.dump(random_selected, f, indent=2)

            class_rows.append(
                eval_classifier_source("recon_all", clf, test_loader, labels, prime_ecg_model, prime_ecg_name, None, args)
            )
            class_rows.append(
                eval_classifier_source(
                    "recon_selected", clf, test_loader, labels, prime_ecg_model, prime_ecg_name, selected, args
                )
            )
            class_rows.append(
                eval_classifier_source(
                    "recon_random_selected",
                    clf,
                    test_loader,
                    labels,
                    prime_ecg_model,
                    prime_ecg_name,
                    random_selected,
                    args,
                )
            )
        save_csv(class_rows, out_dir / "classification_lead_selection_results.csv")

    print("[DONE] PRIME-ECG full experiments complete.")


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    for field, default in Args().__dict__.items():
        if isinstance(default, bool):
            if default:
                p.add_argument(f"--no_{field}", action="store_false", dest=field)
            else:
                p.add_argument(f"--{field}", action="store_true")
        elif isinstance(default, int):
            p.add_argument(f"--{field}", type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(f"--{field}", type=float, default=default)
        else:
            p.add_argument(f"--{field}", default=default)
    return Args(**vars(p.parse_args()))


if __name__ == "__main__":
    run()
