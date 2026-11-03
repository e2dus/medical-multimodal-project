from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


DEFAULT_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]


@dataclass
class UncertaintyBatch:
    mean: torch.Tensor
    epistemic_std: torch.Tensor
    aleatoric_std: Optional[torch.Tensor]
    total_std: torch.Tensor
    samples: torch.Tensor


def _to_numpy(x: TensorLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _extract_prediction(model_out):
    """Return (mean, log_var) from common model output styles.

    Supported outputs:
      pred
      (pred, explain_dict)
      (mean, log_var)
      {"mean": ..., "log_var": ...}
    """
    if isinstance(model_out, torch.Tensor):
        return model_out, None

    if isinstance(model_out, dict):
        mean = model_out.get("mean", model_out.get("pred", model_out.get("prediction")))
        log_var = model_out.get("log_var", model_out.get("logvar", None))
        if mean is None:
            raise ValueError("Dict output must contain 'mean', 'pred', or 'prediction'.")
        return mean, log_var

    if isinstance(model_out, (tuple, list)):
        if len(model_out) == 0:
            raise ValueError("Empty model output.")
        first = model_out[0]
        second = model_out[1] if len(model_out) > 1 else None
        if not isinstance(first, torch.Tensor):
            raise ValueError("First model output must be a tensor prediction.")
        if isinstance(second, torch.Tensor) and second.shape == first.shape:
            return first, second
        return first, None

    raise TypeError(f"Unsupported model output type: {type(model_out)!r}")


def _call_model(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    try:
        out = model(x)
    except TypeError:
        out = model(x, return_explain=False)
    return _extract_prediction(out)


def enable_dropout_only(model: nn.Module) -> None:
    """Keep the model in eval mode but activate dropout layers for MC dropout."""
    model.eval()
    for module in model.modules():
        if isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.AlphaDropout,
                nn.FeatureAlphaDropout,
            ),
        ):
            module.train()


@torch.no_grad()
def predict_mc_dropout(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 100,
    device: Optional[torch.device] = None,
) -> UncertaintyBatch:
    """Repeatedly predict the same input and decompose uncertainty.

    If the model returns (mean, log_var), aleatoric uncertainty is estimated as
    E[exp(log_var)]. If the model returns only pred, aleatoric_std is None and
    total_std equals epistemic_std.
    """
    if device is None:
        device = next(model.parameters()).device
    x = x.to(device)

    was_training = model.training
    enable_dropout_only(model)

    means: List[torch.Tensor] = []
    log_vars: List[torch.Tensor] = []
    for _ in range(n_samples):
        mean, log_var = _call_model(model, x)
        means.append(mean.detach())
        if log_var is not None:
            log_vars.append(log_var.detach())

    if was_training:
        model.train()
    else:
        model.eval()

    samples = torch.stack(means, dim=0)  # [S,B,L,T]
    pred_mean = samples.mean(dim=0)
    epistemic_var = samples.var(dim=0, unbiased=False)
    epistemic_std = torch.sqrt(epistemic_var.clamp_min(0.0))

    if log_vars:
        log_vars_t = torch.stack(log_vars, dim=0)
        aleatoric_var = torch.exp(log_vars_t).mean(dim=0)
        aleatoric_std = torch.sqrt(aleatoric_var.clamp_min(0.0))
        total_std = torch.sqrt((epistemic_var + aleatoric_var).clamp_min(0.0))
    else:
        aleatoric_std = None
        total_std = epistemic_std

    return UncertaintyBatch(
        mean=pred_mean,
        epistemic_std=epistemic_std,
        aleatoric_std=aleatoric_std,
        total_std=total_std,
        samples=samples,
    )


def same_input_different_output_check(
    model: nn.Module,
    x: torch.Tensor,
    n_samples: int = 100,
    lead_names: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    print_result: bool = True,
) -> Dict[str, torch.Tensor]:
    """Check output variability for the exact same input.

    This is the direct test for "same input -> different output".
    With MC dropout enabled, high std means high epistemic uncertainty.
    """
    if lead_names is None:
        lead_names = DEFAULT_LEADS

    result = predict_mc_dropout(model, x, n_samples=n_samples, device=device)
    lead_std = result.epistemic_std.mean(dim=(0, 2)).detach().cpu()
    lead_total = result.total_std.mean(dim=(0, 2)).detach().cpu()

    if print_result:
        print("\n[SAME INPUT -> DIFFERENT OUTPUT]")
        print(f"n_samples = {n_samples}")
        for i, name in enumerate(lead_names[: len(lead_std)]):
            print(f"{name:>4s} | epistemic_std={lead_std[i].item():.6f} | total_std={lead_total[i].item():.6f}")

    return {
        "mean": result.mean.detach().cpu(),
        "epistemic_std": result.epistemic_std.detach().cpu(),
        "aleatoric_std": None if result.aleatoric_std is None else result.aleatoric_std.detach().cpu(),
        "total_std": result.total_std.detach().cpu(),
        "samples": result.samples.detach().cpu(),
        "lead_epistemic_std": lead_std,
        "lead_total_std": lead_total,
    }


@torch.no_grad()
def uncertainty_error_check(
    model: nn.Module,
    dataloader: Iterable,
    n_samples: int = 30,
    lead_names: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    print_result: bool = True,
) -> Dict[str, Union[torch.Tensor, Dict[str, float]]]:
    """Check whether predicted uncertainty matches actual reconstruction error.

    Expected dataloader batch format:
      (x, y) or (x, y, ...)

    Returns lead-wise uncertainty, lead-wise absolute error, and correlations.
    """
    if lead_names is None:
        lead_names = DEFAULT_LEADS
    if device is None:
        device = next(model.parameters()).device

    all_unc: List[torch.Tensor] = []
    all_epi: List[torch.Tensor] = []
    all_err: List[torch.Tensor] = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            x, y = batch[0], batch[1]

        x = x.to(device)
        y = y.to(device)
        result = predict_mc_dropout(model, x, n_samples=n_samples, device=device)
        err = (result.mean - y).abs()

        all_unc.append(result.total_std.detach().cpu())
        all_epi.append(result.epistemic_std.detach().cpu())
        all_err.append(err.detach().cpu())

    total_unc = torch.cat(all_unc, dim=0)
    epistemic_unc = torch.cat(all_epi, dim=0)
    abs_error = torch.cat(all_err, dim=0)

    lead_unc = total_unc.mean(dim=(0, 2))
    lead_epi = epistemic_unc.mean(dim=(0, 2))
    lead_err = abs_error.mean(dim=(0, 2))

    unc_np = _to_numpy(total_unc)
    err_np = _to_numpy(abs_error)
    sample_unc = _to_numpy(total_unc.mean(dim=(1, 2)))
    sample_err = _to_numpy(abs_error.mean(dim=(1, 2)))

    summary = {
        "point_pearson_uncertainty_error": _pearson(unc_np, err_np),
        "point_spearman_uncertainty_error": _spearman(unc_np, err_np),
        "sample_pearson_uncertainty_error": _pearson(sample_unc, sample_err),
        "sample_spearman_uncertainty_error": _spearman(sample_unc, sample_err),
    }

    if print_result:
        print("\n[UNCERTAINTY VS ACTUAL ERROR]")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}")
        print("\nLead-wise mean total uncertainty / epistemic uncertainty / absolute error")
        for i, name in enumerate(lead_names[: len(lead_unc)]):
            print(
                f"{name:>4s} | total_unc={lead_unc[i].item():.6f} "
                f"| epistemic={lead_epi[i].item():.6f} | abs_err={lead_err[i].item():.6f}"
            )

    return {
        "total_uncertainty": total_unc,
        "epistemic_uncertainty": epistemic_unc,
        "absolute_error": abs_error,
        "lead_total_uncertainty": lead_unc,
        "lead_epistemic_uncertainty": lead_epi,
        "lead_absolute_error": lead_err,
        "sample_uncertainty": total_unc.mean(dim=(1, 2)),
        "sample_error": abs_error.mean(dim=(1, 2)),
        "summary": summary,
    }


def uncertainty_vs_diagnosis_accuracy_check(
    uncertainty_score: TensorLike,
    pred_label: TensorLike,
    true_label: TensorLike,
    low_q: float = 0.33,
    high_q: float = 0.67,
    print_result: bool = True,
) -> Dict[str, float]:
    """Check if high-uncertainty samples have lower diagnostic accuracy.

    Args:
        uncertainty_score: one scalar uncertainty per sample, shape [N].
        pred_label: predicted diagnosis class per sample, shape [N].
        true_label: true diagnosis class per sample, shape [N].
    """
    u = _to_numpy(uncertainty_score).reshape(-1)
    pred = _to_numpy(pred_label).reshape(-1)
    true = _to_numpy(true_label).reshape(-1)
    if not (len(u) == len(pred) == len(true)):
        raise ValueError("uncertainty_score, pred_label, and true_label must have the same length.")

    correct = (pred == true).astype(np.float64)
    lo = np.quantile(u, low_q)
    hi = np.quantile(u, high_q)
    low_mask = u <= lo
    high_mask = u >= hi
    mid_mask = (u > lo) & (u < hi)

    def acc(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float(correct[mask].mean())

    out = {
        "low_uncertainty_accuracy": acc(low_mask),
        "mid_uncertainty_accuracy": acc(mid_mask),
        "high_uncertainty_accuracy": acc(high_mask),
        "low_uncertainty_count": int(low_mask.sum()),
        "mid_uncertainty_count": int(mid_mask.sum()),
        "high_uncertainty_count": int(high_mask.sum()),
        "uncertainty_correct_pearson": _pearson(u, correct),
        "uncertainty_correct_spearman": _spearman(u, correct),
    }

    if print_result:
        print("\n[UNCERTAINTY VS DIAGNOSIS ACCURACY]")
        print(f"low uncertainty accuracy : {out['low_uncertainty_accuracy']:.4f} (n={out['low_uncertainty_count']})")
        print(f"mid uncertainty accuracy : {out['mid_uncertainty_accuracy']:.4f} (n={out['mid_uncertainty_count']})")
        print(f"high uncertainty accuracy: {out['high_uncertainty_accuracy']:.4f} (n={out['high_uncertainty_count']})")
        print(f"corr(uncertainty, correct) pearson : {out['uncertainty_correct_pearson']:.4f}")
        print(f"corr(uncertainty, correct) spearman: {out['uncertainty_correct_spearman']:.4f}")

    return out


def save_same_input_plot(
    result: Dict[str, torch.Tensor],
    save_path: Union[str, Path],
    lead_names: Optional[Sequence[str]] = None,
    sample_index: int = 0,
    max_leads: int = 12,
) -> None:
    import matplotlib.pyplot as plt

    if lead_names is None:
        lead_names = DEFAULT_LEADS
    mean = result["mean"][sample_index]
    std = result["total_std"][sample_index]
    n_leads = min(mean.shape[0], max_leads)
    cols = 4
    rows = int(np.ceil(n_leads / cols))
    t = np.arange(mean.shape[-1])

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n_leads:
            ax.axis("off")
            continue
        m = mean[i].numpy()
        s = std[i].numpy()
        ax.plot(t, m, color="tab:blue", linewidth=1.0)
        ax.fill_between(t, m - 2.0 * s, m + 2.0 * s, color="tab:blue", alpha=0.25)
        ax.set_title(str(lead_names[i]))
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_uncertainty_error_plot(
    result: Dict[str, Union[torch.Tensor, Dict[str, float]]],
    save_path: Union[str, Path],
) -> None:
    import matplotlib.pyplot as plt

    u = _to_numpy(result["sample_uncertainty"])
    e = _to_numpy(result["sample_error"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(u, e, s=10, alpha=0.4)
    ax.set_xlabel("Sample uncertainty")
    ax.set_ylabel("Sample absolute reconstruction error")
    ax.set_title("Uncertainty vs actual error")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

