import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import os

# ========================
# Parameter settings
# ========================
INPUT_CSV = "output-1/all_water_coords.csv"
BASE_OUTPUT_DIR = "output-2"

EPS_START = 0.1
EPS_END = 2.0
EPS_STEP = 0.1          #  Change the step size here

MIN_SAMPLES = 25        #  Minimum number of occurrences

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ========================
# Read water coordinates
# ========================
df_all = pd.read_csv(INPUT_CSV)
coords_all = df_all[["x", "y", "z"]].values

print(f"Total water molecules: {len(coords_all)}")

# ========================
# EPS scanning loop
# ========================
eps_values = np.round(
    np.arange(EPS_START, EPS_END + EPS_STEP, EPS_STEP), 3
)

# For saving EPS scan summary
eps_summary = []

for EPS in eps_values:
    print(f"\n=== Running DBSCAN with EPS = {EPS:.2f} Å ===")

    # ------------------------
    # Output directory for current EPS
    # ------------------------
    eps_dir = os.path.join(BASE_OUTPUT_DIR, f"eps_{EPS:.2f}")
    os.makedirs(eps_dir, exist_ok=True)

    # Copy dataframe to avoid contamination
    df = df_all.copy()

    # ------------------------
    # DBSCAN clustering
    # ------------------------
    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(coords_all)
    df["site_label"] = labels

    n_sites = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Identified water sites: {n_sites}")

    # Save to eps_summary
    eps_summary.append({
        "EPS": EPS,
        "n_sites": n_sites
    })

    # ------------------------
    # Statistics for water site
    # ------------------------
    site_records = []
    site_id_map = {}

    site_counter = 1
    for label in sorted(set(labels)):
        if label == -1:
            continue

        site_df = df[df["site_label"] == label]

        center = site_df[["x", "y", "z"]].mean().values
        count = len(site_df)
        n_structures = site_df["structure"].nunique()

        site_name = f"WS{site_counter}"
        site_id_map[label] = site_name

        site_records.append({
            "site_id": site_name,
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "water_count": count,
            "structure_occupancy": n_structures
        })

        site_counter += 1

    summary_df = pd.DataFrame(site_records)
    summary_df.sort_values(
        "structure_occupancy", ascending=False, inplace=True
    )

    # ------------------------
    # Save summary
    # ------------------------
    summary_path = os.path.join(eps_dir, "water_sites_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # ------------------------
    # Save assignments
    # ------------------------
    df["water_site"] = df["site_label"].map(site_id_map)
    df.loc[df["site_label"] == -1, "water_site"] = "Noise"

    assign_path = os.path.join(eps_dir, "water_sites_assignments.csv")
    df.to_csv(assign_path, index=False)

    # ------------------------
    # Output PDB (water site centers)
    # ------------------------
    pdb_path = os.path.join(eps_dir, "water_sites_centers.pdb")

    with open(pdb_path, "w") as f:
        atom_id = 1
        for _, row in summary_df.iterrows():
            f.write(
                "HETATM{:5d}  O   HOH A{:4d}    "
                "{:8.3f}{:8.3f}{:8.3f}  1.00{:6.2f}           O\n".format(
                    atom_id,
                    atom_id,
                    row["center_x"],
                    row["center_y"],
                    row["center_z"],
                    row["structure_occupancy"]
                )
            )
            atom_id += 1

# ------------------------
# Save EPS scan summary
# ------------------------
eps_summary_df = pd.DataFrame(eps_summary)
eps_summary_csv = os.path.join(BASE_OUTPUT_DIR, "dbscan_eps_summary.csv")
eps_summary_df.to_csv(eps_summary_csv, index=False)

print("\n=== ALL EPS SCANS FINISHED ===")
print(f"Results saved under: {BASE_OUTPUT_DIR}")
print(f"EPS summary saved at: {eps_summary_csv}")
