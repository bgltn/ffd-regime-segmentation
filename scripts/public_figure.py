# scripts/public_figure.py

from pathlib import Path
import traceback

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output

import matplotlib.pyplot as plt
import pandas as pd


def pair_label(df: pd.DataFrame) -> pd.Series:
    return df["series"].astype(str) + " | " + df["break"].astype(str) + " | " + df["window_id"].astype(str)


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.astype(str)
         .str.strip()
         .str.lower()
         .isin(["true", "1", "yes", "y"])
    )


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    public_dir = root / "docs" / "assets" / "public"
    fig_dir = root / "docs" / "assets" / "figures"

    accepted_path = public_dir / "accepted_cases.csv"
    estimable_path = public_dir / "estimable_pairs.csv"

    print("=== public_figure.py start ===", flush=True)
    print(f"ROOT       : {root}", flush=True)
    print(f"PUBLIC_DIR : {public_dir}", flush=True)
    print(f"FIG_DIR    : {fig_dir}", flush=True)
    print(f"accepted   : {accepted_path}", flush=True)
    print(f"estimable  : {estimable_path}", flush=True)

    fig_dir.mkdir(parents=True, exist_ok=True)

    if not accepted_path.exists():
        raise FileNotFoundError(f"Missing file: {accepted_path}")
    if not estimable_path.exists():
        raise FileNotFoundError(f"Missing file: {estimable_path}")

    accepted = pd.read_csv(accepted_path)
    estimable = pd.read_csv(estimable_path)

    print(f"accepted shape  : {accepted.shape}", flush=True)
    print(f"estimable shape : {estimable.shape}", flush=True)
    print(f"accepted columns  : {list(accepted.columns)}", flush=True)
    print(f"estimable columns : {list(estimable.columns)}", flush=True)

    if accepted.empty:
        raise ValueError("accepted_cases.csv is empty")
    if estimable.empty:
        raise ValueError("estimable_pairs.csv is empty")

    require_columns(
        accepted,
        ["series", "break", "window_id", "d_global", "d_local", "mse_improvement"],
        "accepted_cases.csv",
    )
    require_columns(
        estimable,
        ["delta_rho", "mse_improvement", "segmented"],
        "estimable_pairs.csv",
    )

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "svg.fonttype": "none",
    })

    # ------------------------------------------------------------
    # Figure 1
    # Out-of-sample MSE improvement across accepted cases
    # ------------------------------------------------------------
    print("Building Figure 1...", flush=True)
    f1 = accepted.sort_values("mse_improvement", ascending=True).copy()
    f1["label"] = pair_label(f1)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(f1))))
    ax.barh(f1["label"], f1["mse_improvement"])
    ax.set_title("Out-of-sample MSE improvement across accepted cases")
    ax.set_xlabel("MSE improvement")
    ax.set_ylabel("")
    plt.tight_layout()

    out1 = fig_dir / "mse_improvement_accepted_cases.svg"
    fig.savefig(out1, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out1}", flush=True)

    # ------------------------------------------------------------
    # Figure 2
    # d_global vs d_local for accepted cases
    # ------------------------------------------------------------
    print("Building Figure 2...", flush=True)
    f2 = accepted.copy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(f2["d_global"], f2["d_local"])
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Baseline and regime-specific admissible orders")
    ax.set_xlabel("d_global")
    ax.set_ylabel("d_local")

    for _, row in f2.iterrows():
        ax.annotate(
            str(row["series"]),
            (row["d_global"], row["d_local"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )

    plt.tight_layout()

    out2 = fig_dir / "global_vs_local_d_accepted_cases.svg"
    fig.savefig(out2, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out2}", flush=True)

    # ------------------------------------------------------------
    # Figure 3
    # OOS gain vs memory preservation
    # ------------------------------------------------------------
    print("Building Figure 3...", flush=True)
    f3 = estimable.copy()
    segmented_bool = to_bool_series(f3["segmented"])

    mask_acc = segmented_bool
    mask_non = ~segmented_bool

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        f3.loc[mask_non, "delta_rho"],
        f3.loc[mask_non, "mse_improvement"],
        label="Not accepted",
        alpha=0.80,
    )

    ax.scatter(
        f3.loc[mask_acc, "delta_rho"],
        f3.loc[mask_acc, "mse_improvement"],
        label="Accepted",
        alpha=0.90,
    )

    ax.axhline(0.05, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle="--", linewidth=1)

    ax.set_title("OOS gain and memory preservation")
    ax.set_xlabel("Delta rho = rho_local - rho_global")
    ax.set_ylabel("MSE improvement")
    ax.legend()

    plt.tight_layout()

    out3 = fig_dir / "oos_vs_memory_preservation_23pairs.svg"
    fig.savefig(out3, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out3}", flush=True)

    print("=== public_figure.py done ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, flush=True)
        traceback.print_exc()
        raise