import os
import pandas as pd
from collections import defaultdict
import numpy as np

# ========== Configuration ==========
ROOT_DIR = r"D:\1-Redmi\Linux\8-BRD4-water\4-MD-hbond-network\6-BD1-water-network\S2"
OUTPUT_DIR = r"D:\1-Redmi\Linux\8-BRD4-water\4-MD-hbond-network"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set of nodes of interest (common node set across all networks)
FOCUS_NODES = {
    'WAT1', 'WAT2', 'WAT3', 'WAT4', 'WAT5', 'WAT6',
    'Y97', 'N135', 'C136', 'M132', 'M105', 'P82', 'P86', 'Q85', 'MOL'
}

# Get all subfolders (16, e.g., 4gpj, 4o70, etc.)
subdirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
print(f"Found {len(subdirs)} subfolders: {subdirs}")

# ========== Data structures ==========
# Edge dictionary: key -> (src, dst) normalized as (node1, node2) with node1 < node2
edge_data = defaultdict(lambda: {
    'src': None,
    'dst': None,
    'edge_type': None,
    'weights': [],
    'counts': [],
    'networks': []
})

# Node dictionary: node_id -> {type, xs, ys, zs, occupancies, water_counts, networks}
node_data = defaultdict(lambda: {
    'type': None,
    'xs': [],
    'ys': [],
    'zs': [],
    'occupancies': [],
    'water_counts': [],
    'networks': []
})

# ========== Iterate over all folders ==========
for sub in subdirs:
    folder_path = os.path.join(ROOT_DIR, sub)
    edges_file = os.path.join(folder_path, 'extracted_edges.csv')
    nodes_file = os.path.join(folder_path, 'extracted_nodes.csv')
    
    if not os.path.isfile(edges_file) or not os.path.isfile(nodes_file):
        print(f"Warning: {sub} missing files, skipped")
        continue
    
    # ---- Read edges ----
    df_edges = pd.read_csv(edges_file)
    # Filter: both src and dst must be in focus nodes
    mask = df_edges['src'].isin(FOCUS_NODES) & df_edges['dst'].isin(FOCUS_NODES)
    df_edges = df_edges[mask].copy()
    
    for _, row in df_edges.iterrows():
        src, dst = row['src'], row['dst']
        # Normalize edge: ensure node1 < node2 (string comparison)
        if src < dst:
            key = (src, dst)
            src_out, dst_out = src, dst
        else:
            key = (dst, src)
            src_out, dst_out = dst, src
        
        edge_dict = edge_data[key]
        if edge_dict['src'] is None:
            edge_dict['src'] = src_out
            edge_dict['dst'] = dst_out
            edge_dict['edge_type'] = row['edge_type']  #  Assume edge type is consistent for same node pair
        # Append weight, count and network name
        edge_dict['weights'].append(row['weight'])
        edge_dict['counts'].append(row['count'])
        edge_dict['networks'].append(sub)
    
    # ---- Read nodes ----
    df_nodes = pd.read_csv(nodes_file)
    # Filter focus nodes
    df_nodes = df_nodes[df_nodes['node_id'].isin(FOCUS_NODES)].copy()
    
    for _, row in df_nodes.iterrows():
        nid = row['node_id']
        nd = node_data[nid]
        if nd['type'] is None:
            nd['type'] = row['type']
        # Append numeric attributes (may be missing, fill with NaN)
        nd['xs'].append(row['x'] if pd.notna(row['x']) else np.nan)
        nd['ys'].append(row['y'] if pd.notna(row['y']) else np.nan)
        nd['zs'].append(row['z'] if pd.notna(row['z']) else np.nan)
        nd['occupancies'].append(row['occupancy'] if pd.notna(row['occupancy']) else np.nan)
        nd['water_counts'].append(row['water_count'] if pd.notna(row['water_count']) else np.nan)
        nd['networks'].append(sub)

# ========== Generate merged edge table ==========
edge_records = []
for (src, dst), data in edge_data.items():
    occ = len(data['networks'])
    cluster = 'all' if occ == len(subdirs) else 'part'
    avg_weight = np.mean(data['weights'])
    avg_count = np.mean(data['counts'])
    # Can also keep sum, but here we take average
    edge_records.append({
        'src': src,
        'dst': dst,
        'edge_type': data['edge_type'],
        'weight': avg_weight,
        'count': avg_count,
        'occurrence': occ,
        'cluster': cluster
    })

df_edges_out = pd.DataFrame(edge_records)
# Sort by cluster, 'all' first, then 'part'
df_edges_out = df_edges_out.sort_values('cluster', ascending=False)  # 'all' > 'part'
out_edges_path = os.path.join(OUTPUT_DIR, 'merged_edges_S2.csv')
df_edges_out.to_csv(out_edges_path, index=False)
print(f"Merged edges output to: {out_edges_path}, total {len(df_edges_out)} edges")

# ========== Generate merged node table ==========
node_records = []
for nid, data in node_data.items():
    # Compute average coordinates, ignoring NaN
    avg_x = np.nanmean(data['xs']) if data['xs'] else np.nan
    avg_y = np.nanmean(data['ys']) if data['ys'] else np.nan
    avg_z = np.nanmean(data['zs']) if data['zs'] else np.nan
    avg_occ = np.nanmean(data['occupancies']) if data['occupancies'] else np.nan
    avg_wc = np.nanmean(data['water_counts']) if data['water_counts'] else np.nan
    occ_networks = len(set(data['networks']))  # Number of networks where this node appears
    node_records.append({
        'node_id': nid,
        'type': data['type'],
        'x': avg_x,
        'y': avg_y,
        'z': avg_z,
        'occupancy': avg_occ,
        'water_count': avg_wc,
        'network_occurrence': occ_networks
    })

# Ensure all focus nodes are included in output (even if not appearing in any edge)
for nid in FOCUS_NODES:
    if nid not in node_data:
        # No data, leave blank
        node_records.append({
            'node_id': nid,
            'type': None,
            'x': np.nan,
            'y': np.nan,
            'z': np.nan,
            'occupancy': np.nan,
            'water_count': np.nan,
            'network_occurrence': 0
        })

df_nodes_out = pd.DataFrame(node_records)
out_nodes_path = os.path.join(OUTPUT_DIR, 'merged_nodes_S2.csv')
df_nodes_out.to_csv(out_nodes_path, index=False)
print(f"Merged nodes output to: {out_nodes_path}, total {len(df_nodes_out)} nodes")

print("Processing complete!")