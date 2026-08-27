import pandas as pd
import os

# =========================================================
# Configuration paths
# =========================================================
ROOT_DIR = r"D:\1-Redmi\Linux\8-BRD4-water\4-MD-hbond-network\6-BD1-water-network"
MAP_FILE = os.path.join(ROOT_DIR, "watersite-number.txt")   #  Mapping file location

# Target node list (residues or ligand)
TARGET_NODES = ['Y97', 'N135', 'C136', 'M132', 'P82', 'Q85', 'P86', 'M105', 'MOL']

# =========================================================
# Read mapping file: pdb → {WS number: WAT label}
# =========================================================
if not os.path.exists(MAP_FILE):
    raise FileNotFoundError(f"Mapping file does not exist:{MAP_FILE}")

map_df = pd.read_csv(MAP_FILE)   # Columns: pdb, WAT2, WAT1, WAT3, WAT4, WAT5, WAT6
pdb_mapping = {}

for _, row in map_df.iterrows():
    pdb = row['pdb']
    mapping = {}
    # Corresponding relationship: WAT2→column 2, WAT1→column 3, WAT3→column 4, WAT4→column 5, WAT5→column 6, WAT6→column 7
    wat_labels = ['WAT2', 'WAT1', 'WAT3', 'WAT4', 'WAT5', 'WAT6']
    for label in wat_labels:
        ws_num = row[label]
        if pd.notna(ws_num) and ws_num != 0:   #  0 means no corresponding label
            mapping[str(int(ws_num))] = label   # WS number → WAT label
    pdb_mapping[pdb] = mapping

print("Loaded mapping for {} PDBs.".format(len(pdb_mapping)))

# =========================================================
# Iterate over subfolders, extract and replace
# =========================================================
for folder in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    pdb = folder
    if pdb not in pdb_mapping:
        print(f"Warning: {pdb} not in mapping file, skipped")
        continue

    nodes_path = os.path.join(folder_path, "graph_nodes.csv")
    edges_path = os.path.join(folder_path, "graph_edges.csv")
    if not (os.path.exists(nodes_path) and os.path.exists(edges_path)):
        print(f"Warning: {pdb} missing graph_nodes.csv or graph_edges.csv, skipped")
        continue

    # Read data
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    # Determine which target nodes actually exist
    existing_targets = [n for n in TARGET_NODES if n in nodes_df['node_id'].values]

    # Find all WaterSite nodes directly connected to target nodes
    watersite_set = set()
    # First check edges to find neighboring nodes
    for _, row in edges_df.iterrows():
        src = row['src']
        dst = row['dst']
        # If one end is a target node and the other is a WaterSite, collect that WaterSite
        if src.startswith('WS') and dst in existing_targets:
            watersite_set.add(src)
        elif dst.startswith('WS') and src in existing_targets:
            watersite_set.add(dst)

    #  Final node set: target nodes + collected WaterSites
    final_nodes = set(existing_targets) | watersite_set

    # Filter edges: both ends must be in final_nodes
    mask = edges_df['src'].isin(final_nodes) & edges_df['dst'].isin(final_nodes)
    filtered_edges = edges_df[mask].copy()

    # Filter nodes
    filtered_nodes = nodes_df[nodes_df['node_id'].isin(final_nodes)].copy()

    # ------------------- Replace WaterSite IDs -------------------
    mapping = pdb_mapping[pdb]   # Mapping dictionary for current PDB {WS number: WAT label}

    def replace_ws_id(node_id):
        if node_id.startswith('WS'):
            num = node_id[2:]          # Remove 'WS'
            if num in mapping:
                return mapping[num]    # Replace with WAT2, WAT1, ...
        return node_id

    filtered_nodes['node_id'] = filtered_nodes['node_id'].apply(replace_ws_id)
    filtered_edges['src'] = filtered_edges['src'].apply(replace_ws_id)
    filtered_edges['dst'] = filtered_edges['dst'].apply(replace_ws_id)

    # Save results
    out_nodes = os.path.join(folder_path, "extracted_nodes.csv")
    out_edges = os.path.join(folder_path, "extracted_edges.csv")
    filtered_nodes.to_csv(out_nodes, index=False)
    filtered_edges.to_csv(out_edges, index=False)

    print(f"Processing complete: {pdb}, nodes {len(filtered_nodes)}, edges {len(filtered_edges)}")

print("\nAll processing completed.")