#!/usr/bin/env python
# -*- coding: utf-8 -*-
import MDAnalysis as mda
import numpy as np
import pandas as pd
import networkx as nx
import os
import sys
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.lib.distances import distance_array
import gc
import traceback
from tqdm import tqdm
import csv

# =========================================================
# 1. Parameters
# =========================================================
topology = "../../tops/4o70.pdbw.prmtop"
trajectory = "../occupancy-nc/4o70.nc"
PROT_SEL = "resid 1:125"
LIG_SEL  = "resname MOL"
WAT_SEL = "resname WAT"
DIST_CUTOFF = 3.0
ANGLE_CUTOFF = 135.0
N_CORES = 10
FRAME_STEP = 1

# Specify the frame range for analysis
START_FRAME = 0
END_FRAME = 5000

# output
FRAME_PDB_DIR = "4o70-frame-hbond-water"
DENSITY_DIR = "4o70_hbond_water_density"
os.makedirs(FRAME_PDB_DIR, exist_ok=True)
os.makedirs(DENSITY_DIR, exist_ok=True)

# =========================================================
# Density parameters
# =========================================================
GRID_SPACING = 1.0
GRID_MARGIN = 5.0
SMOOTH_SIGMA = 1.0

# =========================================================
# 2. Hydrogen bond classification
# =========================================================
def classify(d_res, a_res):
    def type_of(res):
        if res == "WAT":
            return "water"
        elif res == "MOL":
            return "ligand"
        else:
            return "protein"
    d = type_of(d_res)
    a = type_of(a_res)
    if d == "ligand" and a == "water":
        return "ligand-water"
    if d == "water" and a == "ligand":
        return "water-ligand"
    if d == "protein" and a == "ligand":
        return "protein-ligand"
    if d == "ligand" and a == "protein":
        return "ligand-protein"
    if d == "protein" and a == "water":
        return "protein-water"
    if d == "water" and a == "protein":
        return "water-protein"
    if d == "water" and a == "water":
        return "water-water"
    return "other"

def is_target_hbond_type(hbond_type):
    """Determine whether the hydrogen bond type should be kept"""
    target_types = ["ligand-water", "water-ligand", "protein-water", "water-protein"]
    return hbond_type in target_types

# =========================================================
# 3. Write frame PDB
# =========================================================
def write_frame_pdb(frame, water_data, output_dir):
    frame = int(frame)
    filename = os.path.join(output_dir, f"frame_{frame:05d}.pdb")
    if not water_data:
        with open(filename, "w") as f:
            f.write("REMARK   0 No water molecules in this frame\n")
            f.write("END\n")
        return
    atom_id = 1
    with open(filename, "w") as f:
        f.write(f"REMARK   0 Frame {frame} - Filtered water molecules (target H-bond types only)\n")
        f.write(f"REMARK   0 Total waters: {len(water_data)}\n")
        for wat in water_data:
            coord = wat["coord"]
            resid = int(wat["resid"])
            x, y, z = coord[0], coord[1], coord[2]
            f.write(
                f"HETATM{atom_id:5d}  O   WAT W{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00           O  \n"
            )
            atom_id += 1
        f.write("END\n")

def validate_filtered_water_file(filtered_water_file):
    if not os.path.exists(filtered_water_file):
        return False, 0, 0
    try:
        df = pd.read_csv(filtered_water_file)
        required_columns = ["frame", "water_resid", "water_index", "x", "y", "z"]
        for col in required_columns:
            if col not in df.columns:
                return False, 0, 0
        frame_count = df["frame"].nunique()
        water_count = len(df)
        return True, frame_count, water_count
    except Exception:
        return False, 0, 0

# =========================================================
# 4. Helper function: process single-frame hydrogen bond results (for step 1)
# =========================================================
def _process_frame(frame, hbonds, universe, hb_writer, water_writer):
    frame_int = int(frame)
    try:
        universe.trajectory[frame_int]
    except IndexError:
        print(f"ERROR: Frame {frame_int} out of range")
        return
    except TypeError as e:
        print(f"ERROR: frame {frame_int} type error: {e}")
        return

    processed_water_indices = set()
    for h in hbonds:
        frame_idx = int(h[0])
        if frame_idx != frame_int:
            continue
        donor_idx = int(h[1])
        acceptor_idx = int(h[3])
        dist_hb = h[4]
        angle = h[5]
        d_atom = universe.atoms[donor_idx]
        a_atom = universe.atoms[acceptor_idx]
        hb_type = classify(d_atom.resname, a_atom.resname)
        hb_writer.writerow([
            frame_int, d_atom.resname, d_atom.resid, d_atom.name,
            a_atom.resname, a_atom.resid, a_atom.name,
            f"{dist_hb:.6f}", f"{angle:.6f}", hb_type
        ])
        if d_atom.resname == "WAT" and (d_atom.name == "O" or d_atom.name == "OW"):
            idx = d_atom.index
            if idx not in processed_water_indices:
                water_writer.writerow([
                    frame_int, d_atom.resid, idx,
                    f"{d_atom.position[0]:.6f}", f"{d_atom.position[1]:.6f}", f"{d_atom.position[2]:.6f}"
                ])
                processed_water_indices.add(idx)
        if a_atom.resname == "WAT" and (a_atom.name == "O" or a_atom.name == "OW"):
            idx = a_atom.index
            if idx not in processed_water_indices:
                water_writer.writerow([
                    frame_int, a_atom.resid, idx,
                    f"{a_atom.position[0]:.6f}", f"{a_atom.position[1]:.6f}", f"{a_atom.position[2]:.6f}"
                ])
                processed_water_indices.add(idx)

# =========================================================
# 5. Step 1: Analyze hydrogen bonds
# =========================================================
def worker_step1(args):
    start_frame, stop_frame, wid = args
    u = mda.Universe(topology, trajectory)
    hb_file = f"hbonds_part_{wid}.csv"
    water_file = f"water_all_part_{wid}.csv"
    hb_columns = [
        "frame", "donor_res", "donor_resid", "donor_atom",
        "acceptor_res", "acceptor_resid", "acceptor_atom",
        "distance", "angle", "type"
    ]
    water_columns = ["frame", "water_resid", "water_index", "x", "y", "z"]
    hb_fh = open(hb_file, 'w', newline='')
    water_fh = open(water_file, 'w', newline='')
    hb_writer = csv.writer(hb_fh)
    water_writer = csv.writer(water_fh)
    hb_writer.writerow(hb_columns)
    water_writer.writerow(water_columns)
    flush_interval = 100
    print(f"[Worker Step1-{wid}] started frames {start_frame}-{stop_frame-1}")
    try:
        DONOR_SEL = (
            "(protein and (name N* O*)) "
            "or (resname MOL and (name N* O* F*)) "
            "or (resname WAT and name O*)"
        )
        HYDROGEN_SEL = (
            "(protein and (name H*)) "
            "or (resname MOL and name H*) "
            "or (resname WAT and name H*)"
        )
        ACCEPTOR_SEL = (
            "(protein and (name N* O*)) "
            "or (resname MOL and (name N* O* F*)) "
            "or (resname WAT and name O*)"
        )
        hba = HydrogenBondAnalysis(
            universe=u,
            donors_sel=DONOR_SEL,
            hydrogens_sel=HYDROGEN_SEL,
            acceptors_sel=ACCEPTOR_SEL,
            d_a_cutoff=DIST_CUTOFF,
            d_h_a_angle_cutoff=ANGLE_CUTOFF
        )
        hba.run(start=start_frame, stop=stop_frame, step=FRAME_STEP)
        all_hbonds = hba.results.hbonds
        if len(all_hbonds) == 0:
            print(f"[Worker Step1-{wid}] No H-bonds found")
            return hb_file, water_file
        if isinstance(all_hbonds, np.ndarray):
            all_hbonds_list = all_hbonds.tolist()
        else:
            all_hbonds_list = all_hbonds
        all_hbonds_sorted = sorted(all_hbonds_list, key=lambda x: x[0])
        frame_count = 0
        current_frame = None
        frame_hbonds = []
        for hbond in all_hbonds_sorted:
            frame_idx = hbond[0]
            if current_frame is None:
                current_frame = frame_idx
            if frame_idx != current_frame:
                _process_frame(current_frame, frame_hbonds, u, hb_writer, water_writer)
                frame_count += 1
                if frame_count % flush_interval == 0:
                    hb_fh.flush()
                    water_fh.flush()
                current_frame = frame_idx
                frame_hbonds = []
            frame_hbonds.append(hbond)
        if frame_hbonds:
            _process_frame(current_frame, frame_hbonds, u, hb_writer, water_writer)
            frame_count += 1
        hb_fh.flush()
        water_fh.flush()
        print(f"[Worker Step1-{wid}] finished, processed {frame_count} frames")
        del hba
        gc.collect()
    except Exception as e:
        print(f"[Worker Step1-{wid}] FATAL ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        hb_fh.close()
        water_fh.close()
    return hb_file, water_file

# =========================================================
# 6. Steps 2: Filtering (enhanced: includes water-water and ligand-protein)
# =========================================================
def worker_step3_4(args):
    wid, hb_file, water_file = args
    
    water_columns = ["frame", "water_resid", "water_index", "x", "y", "z"]
    hb_columns = [
        "frame", "donor_res", "donor_resid", "donor_atom",
        "acceptor_res", "acceptor_resid", "acceptor_atom",
        "distance", "angle", "type"
    ]
    
    filtered_hb_file = f"hbonds_filtered_part_{wid}.csv"      # Final hydrogen bond file
    filtered_water_file = f"water_filtered_part_{wid}.csv"    # Water file (only contains waters)
    
    print(f"[Worker Step3-4-{wid}] started filtering")
    
    # ---- Pass 1: Collect set 1 (all water molecules involved in part 1) ----
    valid_water_set = set()
    try:
        print(f"[Worker Step3-4-{wid}] Pass 1: Collecting water IDs from target H-bonds...")
        chunk_size = 100000
        for chunk in pd.read_csv(hb_file, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                hb_type = row["type"]
                if is_target_hbond_type(hb_type):
                    frame = int(row["frame"])
                    if row["donor_res"] == "WAT":
                        valid_water_set.add((frame, int(row["donor_resid"])))
                    if row["acceptor_res"] == "WAT":
                        valid_water_set.add((frame, int(row["acceptor_resid"])))
            del chunk
            gc.collect()
        print(f"[Worker Step3-4-{wid}] Collected {len(valid_water_set)} unique water IDs")
    except Exception as e:
        print(f"[Worker Step3-4-{wid}] ERROR in pass 1: {e}", file=sys.stderr)
        traceback.print_exc()
        return 0, 0

    # ---- Pass 2: Filter hydrogen bonds ----
    # Retention rules:
    #   - part 1: ligand-water, water-ligand, protein-water, water-protein → keep directly
    #   - part 3: ligand-protein, protein-ligand → keep directly
    #   - water-water: both waters must be in valid_water_set → keep
    #   - other types discarded
    hb_filtered_count = 0
    try:
        print(f"[Worker Step3-4-{wid}] Pass 2: Filtering H-bonds...")
        with open(filtered_hb_file, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(hb_columns)
            chunk_size = 100000
            for chunk in pd.read_csv(hb_file, chunksize=chunk_size):
                rows_to_write = []
                for _, row in chunk.iterrows():
                    hb_type = row["type"]
                    # part1
                    if is_target_hbond_type(hb_type):
                        rows_to_write.append(row.tolist())
                        hb_filtered_count += 1
                    # part3
                    elif hb_type in ["ligand-protein", "protein-ligand"]:
                        rows_to_write.append(row.tolist())
                        hb_filtered_count += 1
                    # water-water
                    elif hb_type == "water-water":
                        frame = int(row["frame"])
                        donor_resid = int(row["donor_resid"])
                        acceptor_resid = int(row["acceptor_resid"])
                        if (frame, donor_resid) in valid_water_set and (frame, acceptor_resid) in valid_water_set:
                            rows_to_write.append(row.tolist())
                            hb_filtered_count += 1
                    # others ignored
                writer.writerows(rows_to_write)
                out_f.flush()
                del rows_to_write, chunk
                gc.collect()
        print(f"[Worker Step3-4-{wid}] Kept {hb_filtered_count} H-bonds")
    except Exception as e:
        print(f"[Worker Step3-4-{wid}] ERROR in pass 2: {e}", file=sys.stderr)
        traceback.print_exc()
        return 0, 0

    # ---- Pass 3: Extract water coordinates (only from valid_water_set, deduplicated) ----
    water_filtered_count = 0
    written_water_set = set()
    try:
        print(f"[Worker Step3-4-{wid}] Pass 3: Extracting water coordinates...")
        with open(filtered_water_file, 'w', newline='') as out_f:
            writer = csv.writer(out_f)
            writer.writerow(water_columns)
            chunk_size = 100000
            for chunk in pd.read_csv(water_file, chunksize=chunk_size):
                rows_to_write = []
                for _, row in chunk.iterrows():
                    frame = int(row["frame"])
                    water_resid = int(row["water_resid"])
                    key = (frame, water_resid)
                    if key in valid_water_set and key not in written_water_set:
                        rows_to_write.append(row.tolist())
                        water_filtered_count += 1
                        written_water_set.add(key)
                writer.writerows(rows_to_write)
                out_f.flush()
                del rows_to_write, chunk
                gc.collect()
        print(f"[Worker Step3-4-{wid}] Extracted {water_filtered_count} unique water records")
    except Exception as e:
        print(f"[Worker Step3-4-{wid}] ERROR in pass 3: {e}", file=sys.stderr)
        traceback.print_exc()
        return 0, 0

    del valid_water_set, written_water_set
    gc.collect()
    return water_filtered_count, hb_filtered_count

# =========================================================
# 6. Remaining functions (merge_csv_files, generate_density_streaming, generate_pdb_streaming, generate_network_streaming)
# =========================================================
def merge_csv_files(input_files, output_file, columns):
    if not input_files:
        return 0
    count = 0
    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(columns)
        for idx, infile in enumerate(input_files):
            if not os.path.exists(infile):
                continue
            with open(infile, 'r') as in_f:
                reader = csv.reader(in_f)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                for row in reader:
                    writer.writerow(row)
                    count += 1
            out_f.flush()
            if (idx + 1) % 10 == 0:
                print(f"    Merged {idx+1}/{len(input_files)} files")
    return count

def generate_density_streaming(filtered_water_file, density_file, dx_file):
    ref_u = mda.Universe(topology, trajectory)
    ref_u.trajectory[0]
    ref_prot_lig = ref_u.select_atoms(PROT_SEL + " or " + LIG_SEL)
    ref_coords = ref_prot_lig.positions
    min_coord = ref_coords.min(axis=0) - GRID_MARGIN
    max_coord = ref_coords.max(axis=0) + GRID_MARGIN
    grid_shape = np.ceil((max_coord - min_coord) / GRID_SPACING).astype(int)
    density = np.zeros(grid_shape, dtype=np.float32)
    total_points = 0
    chunk_size = 100000
    for chunk in pd.read_csv(filtered_water_file, chunksize=chunk_size):
        coords = chunk[["x", "y", "z"]].values.astype(np.float32)
        for c in coords:
            idx = ((c - min_coord) / GRID_SPACING).astype(int)
            if np.all(idx >= 0) and np.all(idx < grid_shape):
                density[tuple(idx)] += 1
                total_points += 1
        del coords, chunk
        gc.collect()
    print(f"  Total points in density: {total_points}")
    density = gaussian_filter(density, sigma=SMOOTH_SIGMA)
    np.save(density_file, density)
    with open(dx_file, "w") as f:
        f.write(f"object 1 class gridpositions counts {grid_shape[0]} {grid_shape[1]} {grid_shape[2]}\n")
        f.write(f"origin {min_coord[0]} {min_coord[1]} {min_coord[2]}\n")
        f.write(f"delta {GRID_SPACING} 0 0\n")
        f.write(f"delta 0 {GRID_SPACING} 0\n")
        f.write(f"delta 0 0 {GRID_SPACING}\n")
        f.write(f"object 2 class gridconnections counts {grid_shape[0]} {grid_shape[1]} {grid_shape[2]}\n")
        f.write(f"object 3 class array type double rank 0 items {density.size} data follows\n")
        flat = density.flatten()
        for i in range(0, len(flat), 3):
            f.write(" ".join(f"{v:.6f}" for v in flat[i:i+3]) + "\n")
        f.write('attribute "dep" string "positions"\n')
        f.write('object "hbond_water_density" class field\n')
        f.write('component "positions" value 1\n')
        f.write('component "connections" value 2\n')
        f.write('component "data" value 3\n')
    return grid_shape

def generate_pdb_streaming(filtered_water_file, output_dir, verbose=True):
    if not os.path.exists(filtered_water_file):
        print(f"  ERROR: Filtered water file not found: {filtered_water_file}")
        return 0, 0
    pdb_count = 0
    water_count_total = 0
    dtype_spec = {
        "frame": np.int64,
        "water_resid": np.int64,
        "water_index": np.int64,
        "x": np.float64,
        "y": np.float64,
        "z": np.float64
    }
    chunk_size = 100000
    if verbose:
        print(f"  Reading filtered water file: {filtered_water_file}")
    try:
        for chunk_idx, chunk in enumerate(pd.read_csv(filtered_water_file, chunksize=chunk_size, dtype=dtype_spec)):
            if verbose and chunk_idx % 10 == 0:
                print(f"    Processing chunk {chunk_idx+1}...")
            for frame, group in chunk.groupby("frame"):
                water_data = []
                for _, row in group.iterrows():
                    water_data.append({
                        "resid": int(row["water_resid"]),
                        "coord": np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
                    })
                    water_count_total += 1
                write_frame_pdb(int(frame), water_data, output_dir)
                pdb_count += 1
            del chunk
            gc.collect()
        if verbose:
            print(f"  Generated {pdb_count} PDB files")
            print(f"  Total water records written: {water_count_total}")
    except Exception as e:
        print(f"  ERROR in generate_pdb_streaming: {e}", file=sys.stderr)
        traceback.print_exc()
        return 0, 0
    return pdb_count, water_count_total

def generate_network_streaming(hb_filtered_file, output_file):
    edge_count = 0
    chunk_size = 100000
    with open(output_file, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["frame", "node1", "node2", "weight", "type"])
        for chunk in pd.read_csv(hb_filtered_file, chunksize=chunk_size):
            for frame, group in chunk.groupby("frame"):
                G = nx.Graph()
                for _, row in group.iterrows():
                    node1 = f"{row['donor_res']}{row['donor_resid']}"
                    node2 = f"{row['acceptor_res']}{row['acceptor_resid']}"
                    if G.has_edge(node1, node2):
                        G[node1][node2]["weight"] += 1
                    else:
                        G.add_edge(node1, node2, weight=1, type=row["type"])
                for u, v, data in G.edges(data=True):
                    writer.writerow([int(frame), u, v, data["weight"], data["type"]])
                    edge_count += 1
                del G
                gc.collect()
            del chunk
            gc.collect()
            out_f.flush()
    return edge_count

# =========================================================
# 7. Main program
# =========================================================
if __name__ == "__main__":
    u = mda.Universe(topology, trajectory)
    n_frames = len(u.trajectory)
    print(f"Total frames: {n_frames}")
    print(f"Processing every {FRAME_STEP}th frame")

    if END_FRAME is None:
        END_FRAME = n_frames
    START_FRAME = max(0, START_FRAME)
    END_FRAME = min(n_frames, END_FRAME)
    if START_FRAME >= END_FRAME:
        print(f"Error: START_FRAME ({START_FRAME}) >= END_FRAME ({END_FRAME}). No frames to process.")
        sys.exit(1)

    frame_indices = list(range(START_FRAME, END_FRAME, FRAME_STEP))
    n_selected = len(frame_indices)
    print(f"Selected frames: {n_selected} (from {START_FRAME} to {END_FRAME-1} step {FRAME_STEP})")
    if n_selected == 0:
        print("No frames selected. Exiting.")
        sys.exit(0)

    chunk_size = max(1, n_selected // N_CORES)
    tasks = []
    for i in range(N_CORES):
        start_pos = i * chunk_size
        end_pos = (i + 1) * chunk_size if i < N_CORES - 1 else n_selected
        start_frame = frame_indices[start_pos]
        end_frame = frame_indices[end_pos - 1] + 1 if end_pos > start_pos else start_frame
        tasks.append((start_frame, end_frame, i))

    print("\n" + "="*64)
    print("[Step 1] Analyzing hydrogen bonds in parallel (batch frames)...")
    print("="*64)
    with Pool(N_CORES, maxtasksperchild=100) as pool:
        results = []
        for res in tqdm(pool.imap_unordered(worker_step1, tasks), total=len(tasks), desc="Step1: H-bond analysis"):
            results.append(res)
    hb_files = [r[0] for r in results]
    water_files = [r[1] for r in results]

    print("\n" + "="*64)
    print("[Step 3+4] Filtering H-bonds and waters (enhanced)...")
    print("  Included types: ligand-water, water-ligand, protein-water, water-protein,")
    print("                  water-water (both waters in set), ligand-protein, protein-ligand")
    print("="*64)
    filter_tasks = [(wid, hb_files[wid], water_files[wid]) for wid in range(N_CORES)]
    filter_results = []
    with Pool(N_CORES, maxtasksperchild=100) as pool:
        for res in tqdm(pool.imap_unordered(worker_step3_4, filter_tasks),
                        total=len(filter_tasks), desc="Step3+4: Filtering"):
            filter_results.append(res)
    water_filtered_counts = [r[0] for r in filter_results]
    hb_filtered_counts = [r[1] for r in filter_results]
    filtered_water_files = [f"water_filtered_part_{i}.csv" for i in range(N_CORES)]
    filtered_hb_files = [f"hbonds_filtered_part_{i}.csv" for i in range(N_CORES)]
    print(f"  Total filtered water records: {sum(water_filtered_counts)}")
    print(f"  Total filtered H-bonds: {sum(hb_filtered_counts)}")

    print("\n" + "="*64)
    print("[Step 3] Merging CSV3 (filtered waters)...")
    print("="*64)
    water_columns = ["frame", "water_resid", "water_index", "x", "y", "z"]
    filtered_water_count = merge_csv_files(filtered_water_files, "4o70_water_filtered.csv", water_columns)
    print(f"  CSV3 saved: 4o70_water_filtered.csv ({filtered_water_count} records)")

    print("\n" + "="*64)
    print("[Step 4] Merging CSV4 (filtered H-bonds)...")
    print("="*64)
    hb_columns = [
        "frame", "donor_res", "donor_resid", "donor_atom",
        "acceptor_res", "acceptor_resid", "acceptor_atom",
        "distance", "angle", "type"
    ]
    filtered_hb_count = merge_csv_files(filtered_hb_files, "4o70_hbonds_filtered_by_water.csv", hb_columns)
    print(f"  CSV4 saved: 4o70_hbonds_filtered_by_water.csv ({filtered_hb_count} records)")

    print("\n[Cleanup] Removing temporary files...")
    temp_files = hb_files + water_files + filtered_water_files + filtered_hb_files
    for f in temp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass
    print("  Temporary files cleaned")
    gc.collect()

    print("\n" + "="*64)
    print("[Step 5] Generating density map (streaming)...")
    print("="*64)
    density_file = os.path.join(DENSITY_DIR, "hbond_water_density.npy")
    dx_file = os.path.join(DENSITY_DIR, "hbond_water_density.dx")
    grid_shape = generate_density_streaming("4o70_water_filtered.csv", density_file, dx_file)
    print(f"  Density map saved to: {DENSITY_DIR}")
    print(f"  Grid shape: {grid_shape}")
    gc.collect()

    print("\n" + "="*64)
    print("[Step 6] Generating PDB files from filtered water CSV...")
    print("="*64)
    is_valid, total_frames, total_waters = validate_filtered_water_file("4o70_water_filtered.csv")
    if is_valid:
        print(f"  Filtered water file validation:")
        print(f"    - Total frames: {total_frames}")
        print(f"    - Total water records: {total_waters}")
        if total_frames > 0:
            print(f"    - Average waters per frame: {total_waters/total_frames:.2f}")
    else:
        print("  WARNING: Filtered water file validation failed.")
    pdb_count, water_count_total = generate_pdb_streaming("4o70_water_filtered.csv", FRAME_PDB_DIR, verbose=True)
    pdb_files = [f for f in os.listdir(FRAME_PDB_DIR) if f.endswith('.pdb')]
    print(f"\n  PDB generation summary:")
    print(f"    - PDB files generated: {len(pdb_files)}")
    print(f"    - Total water records written: {water_count_total}")
    print(f"    - Output directory: {FRAME_PDB_DIR}")

    print("\n" + "="*64)
    print("[Step 7] Generating hydrogen bond network (streaming)...")
    print("="*64)
    edge_count = generate_network_streaming("4o70_hbonds_filtered_by_water.csv",
                                            "4o70_hbonds_network_edges_filtered.csv")
    print(f"  Network edges saved: {edge_count} records")

    print("\n" + "="*64)
    print("Analysis finished successfully")
    print("="*64)
    print("Output files:")
    print("  [CSV3] 4o70_water_filtered.csv             - Filtered waters (part1 only)")
    print("  [CSV4] 4o70_hbonds_filtered_by_water.csv   - H-bonds (part1 + water-water + part3)")
    print("  [NET]  4o70_hbonds_network_edges_filtered.csv - Network edges")
    print(f"  [PDB]  {FRAME_PDB_DIR}                    - PDB files (water only)")
    print(f"  [DEN]  {DENSITY_DIR}                      - Density map files")
    print("="*64)
    print("\nStatistics:")
    print(f"  Total frames processed: {n_selected}")
    print(f"  Total water records (filtered): {filtered_water_count}")
    print(f"  Total H-bonds (filtered): {filtered_hb_count}")
    print(f"  PDB files generated: {pdb_count}")
    print("="*64)
