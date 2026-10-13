"""
Plot service: generate paper-quality scatter plots using Matplotlib.
Produces SVG/PDF/PNG with proper Chinese font support.
"""

import io
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _setup_chinese_font():
    """Try to find a Chinese font for Matplotlib."""
    chinese_fonts = [
        'PingFang SC', 'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
        'Noto Sans CJK SC', 'Source Han Sans SC', 'Hiragino Sans GB',
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in chinese_fonts:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return name
    plt.rcParams['axes.unicode_minus'] = False
    return None


_font_used = _setup_chinese_font()


def compute_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    n = len(y_true)
    if n < 2 or np.var(y_true) == 0:
        r2 = None
    else:
        r2 = float(r2_score(y_true, y_pred))
    return {
        "r2": r2,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": n,
    }


def create_scatter_plots(
    train_y_true: np.ndarray,
    train_y_pred: np.ndarray,
    test_y_true: np.ndarray,
    test_y_pred: np.ndarray,
    target_name: str = "Target",
    unit: str = "",
    output_dir: Path = None,
    formats: list[str] = None,
) -> dict:
    """Create dual-panel scatter plot (train + test) with 1:1 line.

    Returns dict with file paths for each format.
    """
    if formats is None:
        formats = ["png", "svg", "pdf"]

    train_metrics = compute_metrics_from_predictions(train_y_true, train_y_pred)
    test_metrics = compute_metrics_from_predictions(test_y_true, test_y_pred)

    all_vals = np.concatenate([train_y_true, train_y_pred, test_y_true, test_y_pred])
    vmin = np.min(all_vals)
    vmax = np.max(all_vals)
    margin = (vmax - vmin) * 0.05 if vmax > vmin else 1
    domain = [vmin - margin, vmax + margin]

    unit_label = f" ({unit})" if unit else ""
    x_label = f"真实值{unit_label}"
    y_label = f"预测值{unit_label}"

    def _format_metrics(m):
        r2_str = f"{m['r2']:.4f}" if m['r2'] is not None else "N/A"
        return f"R² = {r2_str}\nRMSE = {m['rmse']:.4f}\nMAE = {m['mae']:.4f}\nn = {m['n']}"

    # Dual panel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.09, 3.54))  # 180mm x 90mm

    for ax, yt, yp, title, metrics, color, marker in [
        (ax1, train_y_true, train_y_pred, "训练集", train_metrics, "#0072B2", "o"),
        (ax2, test_y_true, test_y_pred, "测试集", test_metrics, "#D55E00", "^"),
    ]:
        ax.scatter(yt, yp, c=color, marker=marker, s=12, alpha=0.6, edgecolors="none")
        ax.plot(domain, domain, "k--", linewidth=0.8, alpha=0.5, label="y = x")
        ax.set_xlim(domain)
        ax.set_ylim(domain)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal")

        text = _format_metrics(metrics)
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=7,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", edgecolor="gray", alpha=0.8))

    fig.tight_layout()

    results = {}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            path = output_dir / f"scatter_train_test.{fmt}"
            dpi = 300 if fmt == "png" else None
            fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            results[fmt] = str(path)

    plt.close(fig)
    return results


def create_single_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "预测结果",
    target_name: str = "Target",
    unit: str = "",
    color: str = "#0072B2",
    marker: str = "o",
    output_dir: Path = None,
    formats: list[str] = None,
) -> dict:
    """Create a single scatter plot."""
    if formats is None:
        formats = ["png", "svg"]

    metrics = compute_metrics_from_predictions(y_true, y_pred)
    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = np.min(all_vals), np.max(all_vals)
    margin = (vmax - vmin) * 0.05 if vmax > vmin else 1
    domain = [vmin - margin, vmax + margin]

    unit_label = f" ({unit})" if unit else ""

    fig, ax = plt.subplots(figsize=(3.35, 3.35))  # ~85mm square
    ax.scatter(y_true, y_pred, c=color, marker=marker, s=15, alpha=0.6, edgecolors="none")
    ax.plot(domain, domain, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(domain)
    ax.set_ylim(domain)
    ax.set_xlabel(f"真实值{unit_label}", fontsize=9)
    ax.set_ylabel(f"预测值{unit_label}", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal")

    r2_str = f"{metrics['r2']:.4f}" if metrics['r2'] is not None else "N/A"
    text = f"R² = {r2_str}\nRMSE = {metrics['rmse']:.4f}\nMAE = {metrics['mae']:.4f}\nn = {metrics['n']}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", edgecolor="gray", alpha=0.8))

    fig.tight_layout()

    results = {}
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_")
        for fmt in formats:
            path = output_dir / f"scatter_{safe_title}.{fmt}"
            dpi = 300 if fmt == "png" else None
            fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            results[fmt] = str(path)

    plt.close(fig)
    return results


def create_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "残差图",
    output_dir: Path = None,
) -> str:
    """Create residual vs predicted plot."""
    residuals = y_pred - y_true

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    ax.scatter(y_pred, residuals, c="#2563EB", s=10, alpha=0.5, edgecolors="none")
    ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("预测值", fontsize=9)
    ax.set_ylabel("残差 (预测 - 真实)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    path = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = str(output_dir / "residual_plot.png")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")

    plt.close(fig)
    return path
