"""
plot_academic_metrics.py
========================
Plots reward and Q_mean from a CSV file in a clean, publication-ready format.

Usage:
    python plot_academic_metrics.py my_run.csv
    python plot_academic_metrics.py my_run.csv --save --out paper_figure.png
    python plot_academic_metrics.py my_run.csv --window 100 --save
"""

import sys
import os
import argparse

# ── backend selection ─────────────────────────────────────────────────────────
_has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
if not _has_display:
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_tensor(value):
    """Convert 'tensor(0.9990, device=...)' strings (or plain floats) to float."""
    s = str(value).strip()
    if s.startswith("tensor("):
        inner = s[len("tensor("):-1]
        value_str = inner.split(',')[0].strip()
        return float(value_str)
    return float(s)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "eps" in df.columns:
        df["eps"] = df["eps"].apply(parse_tensor)
    return df


# ── plot config ───────────────────────────────────────────────────────────────

def set_academic_style():
    """Applies a clean, paper-ready style to matplotlib."""
    plt.rcParams.update({
        "font.family": "serif",          # Common for academic papers (matches LaTeX)
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,               # High resolution for publication
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "text.color": "black",
        "lines.linewidth": 1.5,
    })


def build_paper_figure(df: pd.DataFrame, window: int = None) -> plt.Figure:
    set_academic_style()
    
    # Create a 1x2 grid (standard for a full-width two-metric figure in papers)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = df["iteration"].values

    # Determine default window if not provided (e.g., 5% of data points)
    if window is None:
        window = max(5, len(x) // 20)

    # --- Subplot 1: Reward ---
    if "reward" in df.columns:
        ax = axes[0]
        y_rew = df["reward"].values
        ma_rew = pd.Series(y_rew).rolling(window, min_periods=1).mean().values
        
        # Plot raw data in background, moving average in foreground
        ax.plot(x, y_rew, alpha=0.25, color="#1f77b4", label="Raw")
        ax.plot(x, ma_rew, color="#08519c", linewidth=2.0, label=f"MA ({window})")
        
        ax.set_title("Training Reward")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")

    # --- Subplot 2: Q Mean ---
    if "q_mean" in df.columns:
        ax = axes[1]
        y_q = df["q_mean"].values
        ma_q = pd.Series(y_q).rolling(window, min_periods=1).mean().values
        
        ax.plot(x, y_q, alpha=0.25, color="#d62728", label="Raw")
        ax.plot(x, ma_q, color="#a50f15", linewidth=2.0, label=f"MA ({window})")
        
        ax.set_title("Mean Q-Value")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Q-Value")

    # --- Formatting for both axes ---
    for ax in axes:
        # Subtle grid
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, color="gray")
        
        # Remove top and right spines to reduce chart junk (Tufte style)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Clean legend
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot RL metrics for academic papers.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="training_metrics.csv",
        help="Path to the CSV file",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Moving average window size (default: 5% of data length)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the figure as a PNG",
    )
    parser.add_argument(
        "--out",
        default="academic_metrics.png",
        help="Output filename (default: academic_metrics.png)",
    )
    args = parser.parse_args()

    print(f"Loading '{args.csv_file}' …")
    try:
        df = load_csv(args.csv_file)
    except FileNotFoundError:
        sys.exit(f"Error: file '{args.csv_file}' not found.")
    except Exception as exc:
        sys.exit(f"Error reading CSV: {exc}")

    fig = build_paper_figure(df, window=args.window)

    if args.save:
        fig.savefig(args.out)
        print(f"Figure saved to '{args.out}'")

    if _has_display:
        plt.show()


if __name__ == "__main__":
    main()