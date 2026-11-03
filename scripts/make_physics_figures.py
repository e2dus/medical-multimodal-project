# -*- coding: utf-8 -*-
"""Physics consistency 시각화 (Fig8: violation bar, Fig9: accuracy-consistency trade-off)."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 300
rcParams["font.size"] = 11
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

phys = pd.read_csv(os.path.join(HERE, "physics_consistency_results.csv"))
recon = pd.read_csv(os.path.join(HERE, "reconstruction_results.csv"))
recon_main = recon[(recon["past_mode"] == "self") & (recon["ablation"] == "none")]
pcc_map = dict(zip(recon_main["method"], recon_main["pcc_mean"]))

PRETTY = {
    "lead_i": "Lead I", "ecgrecover": "ECGrecover(re)",
    "ecgrecover_official": "ECGrecover(off)", "ecgnet": "ECGNet",
    "ekgan": "EKGAN", "mcma": "MCMA", "prime_ecg": "PRIME-ECG",
    "prime_ecg_meta": "PRIME+meta", "no_physics": "no-physics",
}
def col(m):
    if m in ("prime_ecg", "prime_ecg_meta"): return "#d62728"
    if m == "no_physics": return "#ff9896"
    if m == "ecgrecover_official": return "#2ca02c"
    if m == "lead_i": return "#7f7f7f"
    return "#1f77b4"


# Fig 8 — violation bar (log scale)
def fig8():
    d = phys.sort_values("viol_total")
    names = [PRETTY[m] for m in d["method"]]
    colors = [col(m) for m in d["method"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, d["viol_total"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Physics violation (Einthoven+Goldberger MSE) ↓  — log scale")
    ax.set_title("Electrophysiological consistency of reconstructed 12-lead ECG")
    for b, v in zip(bars, d["viol_total"]):
        ax.text(v * 1.1, b.get_y() + b.get_height()/2, f"{v:.4f}",
                va="center", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_physics_violation.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig8 saved")


# Fig 9 — accuracy vs consistency trade-off (the hero plot)
def fig9():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for _, r in phys.iterrows():
        m = r["method"]
        pcc = pcc_map.get(m, np.nan)
        if np.isnan(pcc):
            continue
        ax.scatter(r["viol_total"], pcc, s=180, color=col(m),
                   edgecolor="black", zorder=3)
        ax.annotate(PRETTY[m], (r["viol_total"], pcc),
                    xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Physics violation ↓ (log scale)  — lower = more physically valid")
    ax.set_ylabel("Reconstruction PCC ↑")
    ax.set_title("Accuracy–consistency trade-off\n(top-left = high accuracy + physically valid = best)")
    # 강조 영역
    ax.axhspan(0.79, 0.82, alpha=0.06, color="green")
    ax.annotate("high-accuracy regime",
                xy=(0.002, 0.80), fontsize=9, color="green", alpha=0.8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_accuracy_consistency.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig9 saved")


# Fig 10 — direct ablation: prime_ecg vs no_physics (per-identity)
def fig10():
    sub = phys.set_index("method")
    ids = ["viol_III", "viol_aVR", "viol_aVL", "viol_aVF"]
    id_labels = ["III=II−I", "aVR", "aVL", "aVF"]
    methods = ["prime_ecg_meta", "no_physics"]
    labmap = {"prime_ecg_meta": "PRIME+meta (physics)", "no_physics": "no-physics"}
    colors = {"prime_ecg_meta": "#d62728", "no_physics": "#ff9896"}
    x = np.arange(len(ids)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(methods):
        vals = [sub.loc[m, k] for k in ids]
        ax.bar(x + (i - 0.5) * w, vals, w, label=labmap[m], color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels(id_labels)
    ax.set_ylabel("Physics violation (MSE) ↓")
    ax.set_title("Per-identity violation: physics vs no-physics (same model family)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig10_ablation_physics.png"), bbox_inches="tight")
    plt.close(fig)
    print("fig10 saved")


if __name__ == "__main__":
    fig8()
    fig9()
    fig10()
    print(f"\n완료: {OUT}")
