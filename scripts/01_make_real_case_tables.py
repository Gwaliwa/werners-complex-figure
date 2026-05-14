from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

peaks = pd.read_csv(DATA / "peaks.csv")
werner = pd.read_csv(DATA / "werner_batch_solutions-2.csv")
targets = pd.read_csv(DATA / "ranked_magnetic_target.csv")
drill = pd.read_csv(DATA / "peak_drillhole_ranked_summary.csv")
euler = pd.read_csv(DATA / "euler_solution.csv")
pso = json.loads((DATA / "pso_results_peak4_real_case.json").read_text())

SELECTED = int(pso["selected_peak_index"])

# Selected Peak 4 summary
rows = []
for name, df in [("peaks", peaks), ("werner", werner), ("target_ranking", targets), ("drillhole_validation", drill)]:
    if "peak_index" in df.columns:
        match = df[pd.to_numeric(df["peak_index"], errors="coerce") == SELECTED]
        if not match.empty:
            rec = match.iloc[0].to_dict()
            rec["source_table"] = name
            rows.append(rec)

summary = pd.DataFrame(rows)
summary.to_csv(OUT / "selected_peak4_real_data_summary.csv", index=False)

# Workflow comparison table based on real recorded outputs
A = pso["workflow_A_decomposition_guided"]
B = pso["workflow_B_direct_unconditioned"]
comparison = pd.DataFrame([
    {
        "workflow": "Workflow B: direct/unconditioned inversion",
        "input_signal": "Original complex magnetic profile",
        "werner_depth_px": A["z0_init_px"],
        "pso_depths_px": f"{B['z0_shallow_px']:.2f}; {B['z0_deep_px']:.2f}",
        "rmse_nT": B["pso_rmse_nT"],
        "interpretation": "Mixed multi-source response; higher misfit; lower stability",
    },
    {
        "workflow": "Workflow A: decomposition-guided inversion",
        "input_signal": "Band-pass isolated component",
        "werner_depth_px": A["z0_init_px"],
        "component_depth_m": A["component_depth_m"],
        "pso_depths_px": f"{A['z0_shallow_px']:.2f}; {A['z0_deep_px']:.2f}",
        "rmse_nT": A["pso_rmse_nT"],
        "interpretation": "More coherent effective source estimate after conditioning",
    },
])
comparison.to_csv(OUT / "workflow_comparison_real_case.csv", index=False)

# Euler summary, not as primary depth result
best_euler = euler.sort_values("rmse").head(10)
best_euler.to_csv(OUT / "euler_low_rmse_candidates.csv", index=False)

print("Saved:")
print(OUT / "selected_peak4_real_data_summary.csv")
print(OUT / "workflow_comparison_real_case.csv")
print(OUT / "euler_low_rmse_candidates.csv")
print("\nWorkflow comparison:")
print(comparison.to_string(index=False))
