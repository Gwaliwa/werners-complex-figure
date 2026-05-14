from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

pso = json.loads((DATA / "pso_results_peak4_real_case.json").read_text())
A = pso["workflow_A_decomposition_guided"]
B = pso["workflow_B_direct_unconditioned"]

# RMSE comparison plot
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
labels = ["Direct\nworkflow", "Decomposition-\nguided"]
values = [B["pso_rmse_nT"], A["pso_rmse_nT"]]
ax.bar(labels, values)
ax.set_ylabel("RMSE (nT)")
ax.set_title("Workflow fit-quality comparison for selected Peak 4")
for i, v in enumerate(values):
    ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(OUT / "figure_workflow_rmse_comparison.png", bbox_inches="tight")
plt.close(fig)

# Depth comparison plot
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
labels = ["Direct shallow", "Direct deep", "Decomp. shallow", "Decomp. deep", "Component depth"]
values = [B["z0_shallow_px"], B["z0_deep_px"], A["z0_shallow_px"], A["z0_deep_px"], A["component_depth_px"]]
ax.bar(labels, values)
ax.set_ylabel("Depth estimate (pixels)")
ax.set_title("Depth-estimate consistency for selected Peak 4")
ax.tick_params(axis="x", rotation=25)
for i, v in enumerate(values):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figure_depth_consistency_comparison.png", bbox_inches="tight")
plt.close(fig)

print("Saved workflow comparison figures to outputs/.")
