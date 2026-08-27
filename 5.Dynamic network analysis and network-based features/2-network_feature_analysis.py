#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch water network analysis
============================
Automatically reads watersite-eps.txt, and for each system executes:
1. Network analysis + merge summary + reclassification
2. Generate four scatter plots
3. Output five PDB files (B-factor represents different scores)
"""

import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import argparse

# ============================================================================
# Global configuration (modify according to actual situation)
# ============================================================================
ROOT_DIR = r"D:\1-Redmi\Linux\8-BRD4-water\4-MD-hbond-network"   # Root directory
EPS_FILE = os.path.join(ROOT_DIR, "watersite-eps.txt")           # System vs eps mapping file
TOTAL_STRUCTURES = 5000                                          # Used for occupancy normalization

# ============================================================================
# Utility functions
# ============================================================================
def extract_site_number(site_str):
    """Extract number from 'WS::WS9' or 'WS9'"""
    m = re.search(r"(\d+)", str(site_str))
    return int(m.group(1)) if m else 0

def format_pdb_resname(water_site):
    """Generate PDB residue name (number, at most 3 digits)"""
    num = str(extract_site_number(water_site))
    if len(num) > 3:
        return num[-3:]
    return num

def read_eps_file(eps_path):
    """Read watersite-eps.txt, return {system: eps} dictionary"""
    eps_dict = {}
    with open(eps_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format like "2oss,0.70"
            parts = line.split(',')
            if len(parts) == 2:
                sys_name = parts[0].strip()
                eps_val = float(parts[1].strip())
                eps_dict[sys_name] = eps_val
    return eps_dict

# ============================================================================
# Core analysis function (executes all modules for one system)
# ============================================================================
def run_single_analysis(system, eps, root_dir=ROOT_DIR, total_structures=TOTAL_STRUCTURES):
    """
    Execute complete network analysis for a single system; all input/output paths are automatically constructed.
    """
    print(f"\n{'='*60}")
    print(f"Processing system: {system}  (eps = {eps})")
    print(f"{'='*60}")

    # ---- Construct input/output paths ----
    graph_dir = os.path.join(root_dir, "4-hbond-network-graph", system)
    node_file = os.path.join(graph_dir, "graph_nodes.csv")
    edge_file = os.path.join(graph_dir, "graph_edges.csv")

    summary_dir = os.path.join(root_dir, "3-watersite", system, f"eps_{eps:.2f}")
    summary_file = os.path.join(summary_dir, "water_sites_summary.csv")

    output_dir = os.path.join(root_dir, "5-water-network-analysis", system)
    os.makedirs(output_dir, exist_ok=True)

    # Check if required files exist
    for f in [node_file, edge_file, summary_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    # ---- Module 1: Network analysis + merge + reclassification ----
    print("\n[Module 1] Running network analysis and merging...")
    nodes_df = pd.read_csv(node_file)
    edges_df = pd.read_csv(edge_file)

    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(
            row["node_id"],
            ntype=row["type"],
            x=row["x"],
            y=row["y"],
            z=row["z"],
            occupancy=row["occupancy"]
        )
    for _, row in edges_df.iterrows():
        G.add_edge(row["src"], row["dst"], weight=float(row["weight"]))

    pagerank = nx.pagerank(G, weight="weight")

    water_nodes = nodes_df[nodes_df["type"] == "WaterSite"]["node_id"].tolist()
    if not water_nodes:
        raise RuntimeError("No WaterSite nodes found in graph_nodes.csv")

    max_pr = max(pagerank[w] for w in water_nodes)
    eps_ = 1e-6
    results = []
    for ws in water_nodes:
        pas = 0.0
        lcs = 0.0
        for nb in G.neighbors(ws):
            w = G[ws][nb]["weight"]
            nb_type = G.nodes[nb]["ntype"]
            if nb_type == "Residue":
                pas += w
            elif nb_type == "Ligand":
                lcs += w
        pr_raw = pagerank[ws]
        pr_norm = pr_raw / max_pr if max_pr > 0 else 0.0
        law_score = pr_norm * pas / (pas + lcs + eps_)
        results.append({
            "water_site": ws,
            "pagerank": pr_raw,
            "pagerank_norm": pr_norm,
            "protein_anchoring_score": pas,
            "ligand_coupling_score": lcs,
            "ligand_aware_water_score": law_score
        })

    df_gnn = pd.DataFrame(results)
    summary = pd.read_csv(summary_file)
    df_gnn["site_id"] = df_gnn["water_site"].str.replace("WS::", "", regex=False)
    merged = df_gnn.merge(summary, on="site_id", how="inner")

    def classify(row):
        if row["ligand_aware_water_score"] >= 0.6:
            return "Conserved"
        elif row["ligand_coupling_score"] > 0:
            return "Ligand-displaceable"
        else:
            return "Intermediate"

    merged["water_class"] = merged.apply(classify, axis=1)

    out_merged = os.path.join(output_dir, "water_gnn_with_coordinates.csv")
    merged.to_csv(out_merged, index=False)
    print(f"   Merged file saved: {out_merged} (total {len(merged)} sites)")

    # ---- Module 2: Four scatter plots ----
    print("\n[Module 2] Generating scatter plots...")
    merged["occ_norm"] = merged["structure_occupancy"] / total_structures

    plots = [
        ("occ_norm", "pagerank_norm",
         "Structure Occupancy", "PageRank Score",
         "occupancy_vs_pagerank"),
        ("occ_norm", "ligand_coupling_score",
         "Structure Occupancy", "Ligand Coupling Score",
         "occupancy_vs_ligand_score"),   
        ("occ_norm", "protein_anchoring_score",
         "Structure Occupancy", "Protein Anchoring Score",
         "occupancy_vs_protein_score"), 
    ]

    for xcol, ycol, xlab, ylab, fname in plots:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(merged[xcol], merged[ycol], s=80, alpha=0.8)
        ax.axvline(merged[xcol].median(), linestyle="--", linewidth=2, alpha=0.7)
        ax.axhline(merged[ycol].median(), linestyle="--", linewidth=2, alpha=0.7)
        ax.set_xlabel(xlab, fontsize=24, labelpad=12)
        ax.set_ylabel(ylab, fontsize=24, labelpad=12)
        ax.tick_params(axis='both', which='major', labelsize=22,
                       width=2, length=8, direction='in')
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.tight_layout()
        out_png = os.path.join(output_dir, f"{fname}.png")
        out_pdf = os.path.join(output_dir, f"{fname}.pdf")
        plt.savefig(out_png, dpi=600, bbox_inches="tight")
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved {out_png} and {out_pdf}")

    # ---- Module 3: Output five PDB files (no filtering, no sorting) ----
    print("\n[Module 3] Generating five PDB files with different B-factor scores...")
    # Occupancy normalization already computed (merged["occ_norm"])
    pdb_configs = [
        ("pagerank_norm.pdb", "pagerank_norm"),
        ("water_protein_anchoring.pdb", "protein_anchoring_score"),
        ("water_ligand_coupling.pdb", "ligand_coupling_score"),
        ("water_ligand_aware.pdb", "ligand_aware_water_score"),
        ("water_structure_occupancy.pdb", "occ_norm"),  # using normalized occupancy
    ]

    def write_pdb(df, score_col, out_pdb):
        """Write each site in df as PDB HETATM format, B-factor = score_col"""
        with open(out_pdb, "w") as f:
            atom_id = 1
            for _, row in df.iterrows():
                x = float(row["center_x"])
                y = float(row["center_y"])
                z = float(row["center_z"])
                bfac = float(row[score_col])
                resname = format_pdb_resname(row["water_site"])
                resseq = extract_site_number(row["water_site"])
                f.write(
                    f"HETATM{atom_id:5d}  O   {resname:>3s} W{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{1.00:6.2f}{bfac:6.2f}           O\n"
                )
                atom_id += 1
            f.write("END\n")

    for pdb_file, score_col in pdb_configs:
        out_pdb = os.path.join(output_dir, pdb_file)
        write_pdb(merged, score_col, out_pdb)
        print(f"   Generated {out_pdb} (B-factor = {score_col})")

    print(f"\nFinished processing {system}")

# ============================================================================
# Batch processing main function
# ============================================================================
def batch_run(eps_file=EPS_FILE, root_dir=ROOT_DIR):
    if not os.path.exists(eps_file):
        raise FileNotFoundError(f"EPS file not found: {eps_file}")

    eps_dict = read_eps_file(eps_file)
    print(f"Found {len(eps_dict)} systems: {list(eps_dict.keys())}")

    for system, eps in eps_dict.items():
        try:
            run_single_analysis(system, eps, root_dir=root_dir)
        except Exception as e:
            print(f"\n[ERROR] Failed to process {system}: {e}")
            # Optionally continue to next system

# ============================================================================
# Command-line entry (supports single system or batch)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch water network analysis")
    parser.add_argument("--system", type=str, help="Specify a single system name (e.g., 2oss)")
    parser.add_argument("--eps", type=float, help="Specify eps value (use with --system)")
    parser.add_argument("--batch", action="store_true", help="Process all systems in batch (reads watersite-eps.txt)")
    args = parser.parse_args()

    if args.system and args.eps is not None:
        # Single system mode
        run_single_analysis(args.system, args.eps)
    elif args.batch:
        # Batch mode
        batch_run()
    else:
        # Default: batch mode (if no arguments provided)
        print("No arguments specified, defaulting to batch processing mode...")
        batch_run()