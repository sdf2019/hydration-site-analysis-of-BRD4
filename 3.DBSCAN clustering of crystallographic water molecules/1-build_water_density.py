import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.ndimage import gaussian_filter

# ========================
# Parameter settings
# ========================
PDB_DIR = "input_pdb"
REFERENCE_PDB = "BD1-ref.pdb"
OUTPUT_DIR = "output-1"

GRID_SPACING = 1.0      # Å
GRID_MARGIN = 5.0       # Margin around the reference protein
SMOOTH_SIGMA = 1.0      # Gaussian smoothing sigma σ（Å）

WATER_NAMES = {"HOH", "WAT", "TIP3"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# 工具函数
# ========================
def extract_water_atoms(structure):
    """Extract O atoms of crystal waters"""
    waters = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in WATER_NAMES:
                    if "O" in residue:
                        waters.append(residue["O"])
    return waters


def get_protein_coords(structure):
    """Get all protein atom coordinates (to define grid extent)"""
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    for atom in residue:
                        coords.append(atom.coord)
    return np.array(coords)


# ========================
# Read reference structure
# ========================
parser = PDBParser(QUIET=True)
ref_structure = parser.get_structure("REF", REFERENCE_PDB)

ref_coords = get_protein_coords(ref_structure)
min_coord = ref_coords.min(axis=0) - GRID_MARGIN
max_coord = ref_coords.max(axis=0) + GRID_MARGIN

grid_shape = np.ceil((max_coord - min_coord) / GRID_SPACING).astype(int)
density_grid = np.zeros(grid_shape, dtype=np.float32)

all_water_records = []

# ========================
# Main loop: directly count water coordinates
# ========================
for pdb_file in sorted(os.listdir(PDB_DIR)):
    if not pdb_file.endswith(".pdb"):
        continue

    pdb_path = os.path.join(PDB_DIR, pdb_file)
    structure = parser.get_structure(pdb_file, pdb_path)

    water_atoms = extract_water_atoms(structure)

    for wat in water_atoms:
        coord = wat.coord

        # Record water coordinate
        all_water_records.append({
            "structure": pdb_file,
            "x": coord[0],
            "y": coord[1],
            "z": coord[2]
        })

        # Map to density grid
        idx = ((coord - min_coord) / GRID_SPACING).astype(int)

        if np.all(idx >= 0) and np.all(idx < grid_shape):
            density_grid[tuple(idx)] += 1

    print(f"[OK] {pdb_file}: {len(water_atoms)} waters")

# ========================
# Gaussian smoothing
# ========================
density_grid = gaussian_filter(density_grid, sigma=SMOOTH_SIGMA)

# ========================
# Save results
# ========================
# 1. All water coordinates
df = pd.DataFrame(all_water_records)
df.to_csv(os.path.join(OUTPUT_DIR, "all_water_coords.csv"), index=False)

# 2. density grid
np.save(os.path.join(OUTPUT_DIR, "water_density.npy"), density_grid)

# ========================
# 3. output OpenDX file
# ========================
dx_path = os.path.join(OUTPUT_DIR, "water_density.dx")
with open(dx_path, "w") as f:
    f.write(
        f"object 1 class gridpositions counts "
        f"{grid_shape[0]} {grid_shape[1]} {grid_shape[2]}\n"
    )
    f.write(f"origin {min_coord[0]} {min_coord[1]} {min_coord[2]}\n")
    f.write(f"delta {GRID_SPACING} 0 0\n")
    f.write(f"delta 0 {GRID_SPACING} 0\n")
    f.write(f"delta 0 0 {GRID_SPACING}\n")
    f.write(
        f"object 2 class gridconnections counts "
        f"{grid_shape[0]} {grid_shape[1]} {grid_shape[2]}\n"
    )
    f.write(
        f"object 3 class array type double rank 0 items "
        f"{density_grid.size} data follows\n"
    )

    flat = density_grid.flatten()
    for i in range(0, len(flat), 3):
        f.write(" ".join(f"{v:.6f}" for v in flat[i:i+3]) + "\n")

    f.write("attribute \"dep\" string \"positions\"\n")
    f.write("object \"water_density\" class field\n")
    f.write("component \"positions\" value 1\n")
    f.write("component \"connections\" value 2\n")
    f.write("component \"data\" value 3\n")

print("\n=== WATER DENSITY MAP GENERATED ===")
print("Output files:")
print(" - all_water_coords.csv")
print(" - water_density.npy")
print(" - water_density.dx")
