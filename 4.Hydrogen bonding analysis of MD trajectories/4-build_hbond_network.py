#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build water-site level hydrogen bond interaction network (including multiple edge types)
Input:
    - hbond-network/7wjs_hbonds_filtered_by_water.csv: all hydrogen bond information
    - watersite/7wjs/eps_0.60/water_sites_assignments.csv: water_site assignment for each water molecule
    - watersite/7wjs/eps_0.60/water_sites_summary.csv: statistics for each water_site (center coordinates, number of waters, occupied frames)
Output:
    - hbond_network_graph/7wjs/graph_nodes.csv: node list (WaterSite, Residue, Ligand)
    - hbond_network_graph/7wjs/graph_edges.csv: edge list (WR/WL/WW/PL), weights normalized
"""

import pandas as pd
import os
from collections import defaultdict

# =========================================================
# Input file paths
# =========================================================
HBONDS_CSV = "hbond-network/4o70_hbonds_filtered_by_water.csv"
ASSIGN_CSV  = "watersite/4o70/eps_0.40/water_sites_assignments.csv"
SUMMARY_CSV = "watersite/4o70/eps_0.40/water_sites_summary.csv"
OUTPUT_DIR  = "hbond_network_graph/4o70"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Helper function: three-letter residue code to one-letter code
# =========================================================
RESIDUE_THREE_TO_ONE = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
    'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F',
    'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V',
    # Common modified residues
    'HIE':'H', 'HID':'H', 'HIP':'H', 'ASH':'D', 'GLH':'E', 'LYN':'K',
    'CYM':'C', 'CYX':'C', 'SEC':'U', 'PYL':'O', 'MSE':'M'
}

def three_to_one(resname):
    """"Convert three-letter residue name to one-letter code; if unknown, return original and issue a warning"""
    if resname in RESIDUE_THREE_TO_ONE:
        return RESIDUE_THREE_TO_ONE[resname]
    else:
        # Warning and keep original (or could take first letter)）
        print(f"Warning: Unknown residue name '{resname}', keeping as-is")
        return resname

def get_residue_node(res_name, res_id):
    """
    Generate protein residue node ID, format: single-letter residue code + original residue number. e.g. HIE,396 -> H396
    """
    one_letter = three_to_one(res_name)
    return f"{one_letter}{res_id + 41}"

# =========================================================
# Read data
# =========================================================
print("Reading input files...")
hbonds_df  = pd.read_csv(HBONDS_CSV)
assign_df  = pd.read_csv(ASSIGN_CSV)
summary_df = pd.read_csv(SUMMARY_CSV)

# Ensure correct data types
assign_df["frame"]       = assign_df["frame"].astype(int)
assign_df["water_resid"] = assign_df["water_resid"].astype(int)
hbonds_df["frame"]       = hbonds_df["frame"].astype(int)

# Total number of frames (used for normalizing all edges)
total_frames = hbonds_df["frame"].nunique()
print(f"Total frames:{total_frames}")

# 1. Build (frame, water_resid) -> water_site mapping (excluding Noise)
assign_filtered = assign_df[assign_df["water_site"] != "Noise"].copy()
water_to_site = dict(zip(
    zip(assign_filtered["frame"], assign_filtered["water_resid"]),
    assign_filtered["water_site"]
))

# 2. Read water_site properties (center coordinates, water count, occupancy)
site_info = {}
for _, row in summary_df.iterrows():
    sid = row["site_id"]
    site_info[sid] = {
        "x": row["center_x"],
        "y": row["center_y"],
        "z": row["center_z"],
        "water_count": row["water_count"],
        "occupancy": row["structure_occupancy"]
    }

# =========================================================
# Initialize graph data structures
# =========================================================
nodes = {}
# Use edge_data to store edge counts and types
edge_data = {}  # key: (src, dst), value: {"count": int, "type": str}

# 3. Add all WaterSite nodes (node ID directly uses site_id, e.g., "WS10")
print("Registering WaterSite nodes...")
for sid, info in site_info.items():
    node_id = sid  # e.g., "WS10"
    nodes[node_id] = {
        "type": "WaterSite",
        "x": info["x"],
        "y": info["y"],
        "z": info["z"],
        "occupancy": info["occupancy"],
        "water_count": info["water_count"]
    }

# Pre-register ligand node (node ID = "MOL")
nodes["MOL"] = {"type": "Ligand"}

# Helper function: get water_site for a water molecule
def get_water_site(frame, water_resid):
    key = (frame, water_resid)
    return water_to_site.get(key, None)  # None means Noise

# =========================================================
# Iterate over hydrogen bonds and build edges (handle all types)
# =========================================================
print("Processing hydrogen bond data, accumulating edge counts...")
skipped_noise = 0
skipped_other = 0

for _, row in hbonds_df.iterrows():
    frame = int(row["frame"])
    donor_res   = row["donor_res"]
    donor_resid = int(row["donor_resid"])
    acceptor_res   = row["acceptor_res"]
    acceptor_resid = int(row["acceptor_resid"])
    hb_type = row["type"]

    src_node = None
    dst_node = None
    edge_type = None

    # ---- 1) water-water ----
    if hb_type == "water-water":
        ws1 = get_water_site(frame, donor_resid)
        ws2 = get_water_site(frame, acceptor_resid)
        if ws1 is None or ws2 is None:
            skipped_noise += 1
            continue
        # Normalize direction (smaller ID first)
        if ws1 > ws2:
            ws1, ws2 = ws2, ws1
        src_node = ws1          # directly use site_id, e.g., "WS10"
        dst_node = ws2
        edge_type = "WW"

    # ---- 2) protein-water / water-protein ----
    elif hb_type in ("protein-water", "water-protein"):
        if donor_res == "WAT" and acceptor_res != "WAT":
            water_resid = donor_resid
            prot_res = acceptor_res
            prot_resid = acceptor_resid
        elif acceptor_res == "WAT" and donor_res != "WAT":
            water_resid = acceptor_resid
            prot_res = donor_res
            prot_resid = donor_resid
        else:
            skipped_other += 1
            continue
        ws = get_water_site(frame, water_resid)
        if ws is None:
            skipped_noise += 1
            continue
        src_node = ws
        dst_node = get_residue_node(prot_res, prot_resid)
        edge_type = "WR"

    # ---- 3) ligand-water / water-ligand ----
    elif hb_type in ("ligand-water", "water-ligand"):
        if donor_res == "WAT" and acceptor_res != "WAT":
            water_resid = donor_resid
        elif acceptor_res == "WAT" and donor_res != "WAT":
            water_resid = acceptor_resid
        else:
            skipped_other += 1
            continue
        ws = get_water_site(frame, water_resid)
        if ws is None:
            skipped_noise += 1
            continue
        src_node = ws
        dst_node = "MOL"
        edge_type = "WL"

    # ---- 4) protein-ligand / ligand-protein ----
    elif hb_type in ("protein-ligand", "ligand-protein"):
        if donor_res != "WAT" and acceptor_res != "WAT":
            # Determine the protein residue (the other is ligand)
            if donor_res == "MOL":
                prot_res = acceptor_res
                prot_resid = acceptor_resid
            else:
                prot_res = donor_res
                prot_resid = donor_resid
            src_node = get_residue_node(prot_res, prot_resid)
            dst_node = "MOL"
            edge_type = "PL"
        else:
            skipped_other += 1
            continue

    else:
        skipped_other += 1
        continue

    # Ensure nodes are registered (for dynamically appearing RES and LIG nodes)
    for node in (src_node, dst_node):
        if node not in nodes:
            if node == "MOL":
                nodes[node] = {"type": "Ligand"}
            else:
                # Assume non-"MOL" and non-WaterSite nodes are protein residues (single-letter + number)
                nodes[node] = {"type": "Residue"}

    # Accumulate edge count and record edge type
    key = (src_node, dst_node)
    if key not in edge_data:
        edge_data[key] = {"count": 1, "type": edge_type}
    else:
        edge_data[key]["count"] += 1
        # If types differ, keep the original (should not happen in practice)

print(f"Skipped hydrogen bonds involving Noise waters:{skipped_noise}")
print(f"Skipped other hydrogen bond types:{skipped_other}")

# =========================================================
# Normalize weights and generate edge list (all edges divided by total frames)
# =========================================================
print("Normalizing edge weights...")
edge_rows = []

for (src, dst), data in edge_data.items():
    count = data["count"]
    edge_type = data["type"]
    # All edge weights = event count / total frames
    weight = count / total_frames

    edge_rows.append({
        "src": src,
        "dst": dst,
        "weight": weight,
        "count": count,
        "edge_type": edge_type
    })

# =========================================================
# Output results
# =========================================================
print("Saving node and edge files...")

nodes_df = pd.DataFrame([
    {"node_id": k, **v} for k, v in nodes.items()
])
nodes_df.to_csv(os.path.join(OUTPUT_DIR, "graph_nodes.csv"), index=False)

edges_df = pd.DataFrame(edge_rows)
edges_df.to_csv(os.path.join(OUTPUT_DIR, "graph_edges.csv"), index=False)

print(f"\nDone!")
print(f"Number of nodes:{len(nodes)}")
print(f"Number of edges (after deduplication):{len(edge_rows)}")
print(f"Output directory:{OUTPUT_DIR}")
