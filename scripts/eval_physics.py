# -*- coding: utf-8 -*-
"""
Physical Consistency Evaluation.

전기생리학적 lead 항등식을 각 method의 출력이 얼마나 만족하는지 측정.
물리 모델(prime_ecg)은 공식을 강제하므로 violation≈0,
no_physics / 다른 baseline은 위반 → physics의 임상적 가치 증명.

항등식 (12-lead ECG의 정의):
    III  = II - I              (Einthoven)
    aVR  = -(I + II) / 2        (Goldberger)
    aVL  =  I - II / 2
    aVF  =  II - I / 2

사용법 (서버):
    python eval_physics.py \
        --data_dir /workspace/physionet.org/files/ptb-xl/1.0.3/ \
        --pairs_dir ~/project_jyu/outputs/prime_ecg_meta_500hz \
        --ckpt_dirs ~/project_jyu/outputs/baselines_500hz ~/project_jyu/outputs/prime_ecg_meta_500hz \
        --output_csv ~/project_jyu/outputs/physics_consistency_results.csv \
        --use_hr
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from prime_ecg_full_experiments import (
    Args, IDX, make_model, make_loader, compute_meta_stats,
    model_uses_prior, model_uses_meta,
)

try:
    from prime_ecg_full_experiments import is_tf_recon_method, make_tf_model
except Exception:
    def is_tf_recon_method(name: str) -> bool:
        return False

    def make_tf_model(name: str, args: Args):
        raise RuntimeError("This prime_ecg_full_experiments.py does not include TensorFlow EKGAN support.")


def physics_violation(pred: torch.Tensor):
    """pred: (B, 12, T) -> dict of per-identity MSE violations."""
    I = pred[:, IDX["I"]]
    II = pred[:, IDX["II"]]
    III = pred[:, IDX["III"]]
    aVR = pred[:, IDX["aVR"]]
    aVL = pred[:, IDX["aVL"]]
    aVF = pred[:, IDX["aVF"]]
    v_III = ((III - (II - I)) ** 2).mean(dim=1)            # (B,)
    v_aVR = ((aVR - (-(I + II) / 2.0)) ** 2).mean(dim=1)
    v_aVL = ((aVL - (I - II / 2.0)) ** 2).mean(dim=1)
    v_aVF = ((aVF - (II - I / 2.0)) ** 2).mean(dim=1)
    return v_III, v_aVR, v_aVL, v_aVF


@torch.no_grad()
def eval_method(name, model, loader, device):
    if is_tf_recon_method(name):
        return eval_tf_method(name, model, loader)

    model.eval().to(device)
    accum = {"III": 0.0, "aVR": 0.0, "aVL": 0.0, "aVF": 0.0}
    n = 0
    for batch in tqdm(loader, desc=f"physics {name}", leave=False):
        x = batch["lead_i"].to(device)
        past = batch["past"].to(device) if model_uses_prior(name) else None
        td = batch["time_delta"].to(device) if model_uses_prior(name) else None
        meta = batch["meta"].to(device) if model_uses_meta(name) else None
        out = model(x, past_12lead=past, time_delta=td, patient_meta=meta)
        if isinstance(out, tuple):
            out = out[0]
        vIII, vAVR, vAVL, vAVF = physics_violation(out)
        b = x.size(0)
        accum["III"] += float(vIII.sum().cpu())
        accum["aVR"] += float(vAVR.sum().cpu())
        accum["aVL"] += float(vAVL.sum().cpu())
        accum["aVF"] += float(vAVF.sum().cpu())
        n += b
    res = {f"viol_{k}": v / max(1, n) for k, v in accum.items()}
    res["viol_total"] = sum(res.values())
    res["method"] = name
    return res


@torch.no_grad()
def eval_tf_method(name, model, loader):
    accum = {"III": 0.0, "aVR": 0.0, "aVL": 0.0, "aVF": 0.0}
    n = 0
    for batch in tqdm(loader, desc=f"physics {name}", leave=False):
        out = model.predict_torch_batch(batch)
        vIII, vAVR, vAVL, vAVF = physics_violation(out)
        b = out.size(0)
        accum["III"] += float(vIII.sum().cpu())
        accum["aVR"] += float(vAVR.sum().cpu())
        accum["aVL"] += float(vAVL.sum().cpu())
        accum["aVF"] += float(vAVF.sum().cpu())
        n += b
    res = {f"viol_{k}": v / max(1, n) for k, v in accum.items()}
    res["viol_total"] = sum(res.values())
    res["method"] = name
    return res


def find_checkpoint(name: str, ckpt_dirs):
    """Return (kind, path) for a method checkpoint across fragmented output dirs."""
    for root in ckpt_dirs:
        root = Path(root).expanduser()
        if is_tf_recon_method(name):
            tf_dir = root / f"{name}_best_tf"
            if tf_dir.is_dir():
                return "tf", tf_dir
        pt_file = root / f"{name}_best.pt"
        if pt_file.is_file():
            return "torch", pt_file
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default=None, help="Backward-compatible alias for --pairs_dir and single ckpt dir.")
    ap.add_argument("--pairs_dir", default=None, help="Folder containing train_pairs.csv/test_pairs.csv.")
    ap.add_argument("--ckpt_dirs", nargs="*", default=None, help="One or more folders containing *_best.pt or *_best_tf.")
    ap.add_argument("--output_csv", default=None, help="Where to save physics_consistency_results.csv.")
    ap.add_argument("--methods", default="lead_i,ecgrecover,ecgrecover_official,ecgnet,ekgan,mcma,prime_ecg,prime_ecg_meta,no_physics")
    ap.add_argument("--use_hr", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    cli = ap.parse_args()

    if cli.pairs_dir is None and cli.output_dir is None:
        raise SystemExit("Provide --pairs_dir, or use legacy --output_dir.")

    pairs_dir = Path(cli.pairs_dir or cli.output_dir).expanduser()
    ckpt_dirs = [Path(p).expanduser() for p in (cli.ckpt_dirs or [cli.output_dir or cli.pairs_dir])]
    output_csv = Path(cli.output_csv).expanduser() if cli.output_csv else pairs_dir / "physics_consistency_results.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 저장된 split 재사용
    train_pairs = pd.read_csv(pairs_dir / "train_pairs.csv")
    test_pairs = pd.read_csv(pairs_dir / "test_pairs.csv")
    labels = ["NORM", "MI", "STTC", "CD", "HYP"]
    meta_stats = compute_meta_stats(train_pairs)

    args = Args(
        data_dir=cli.data_dir, use_hr=cli.use_hr,
        batch_size=cli.batch_size, num_workers=cli.num_workers,
    )

    loader = make_loader(Path(cli.data_dir), test_pairs, train_pairs,
                         labels, meta_stats, args, train=False)

    candidates = [x.strip() for x in cli.methods.split(",") if x.strip()]
    rows = []
    for name in candidates:
        kind, ckpt = find_checkpoint(name, ckpt_dirs)
        if ckpt is None:
            searched = ", ".join(str(p) for p in ckpt_dirs)
            print(f"[skip] {name}: checkpoint not found in {searched}")
            continue

        if kind == "tf":
            model = make_tf_model(name, args)
            model.load(ckpt)
        else:
            model = make_model(name, args)
            sd = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(sd, strict=False)

        res = eval_method(name, model, loader, device)
        print(f"{name:22s} total_violation={res['viol_total']:.6f} "
              f"(III={res['viol_III']:.5f} aVR={res['viol_aVR']:.5f} "
              f"aVL={res['viol_aVL']:.5f} aVF={res['viol_aVF']:.5f})")
        rows.append(res)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nsaved: {output_csv}")


if __name__ == "__main__":
    main()
