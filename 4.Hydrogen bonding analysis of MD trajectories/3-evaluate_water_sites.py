import os
import pandas as pd

def process_directory(base_dir, eps_values):
    """
    Process water_sites_summary.csv under a single base_dir (containing eps_* subfolders),
    calculate max, min, total number of sites, number and percentage of sites > 1 for each eps,
    and save a summary CSV.
    """
    # Generate eps string list with two decimal places
    eps_strs = [f'{v:.2f}' for v in eps_values]
    
    results = []
    
    for eps_str in eps_strs:
        subdir = os.path.join(base_dir, f'eps_{eps_str}')
        csv_path = os.path.join(subdir, 'water_sites_summary.csv')
        
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} does not exist, skipped")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            if 'water_count' not in df.columns or 'structure_occupancy' not in df.columns:
                print(f"  Error: {csv_path} is missing 'water_count' or 'structure_occupancy' columns")
                continue
            
            # Calculate water per site
            df['water_per_site'] = df['water_count'] / df['structure_occupancy']
            
            # Statistical results 
            max_wps = df['water_per_site'].max()
            min_wps = df['water_per_site'].min()
            total_sites = len(df)
            num_gt_1 = (df['water_per_site'] > 1).sum()
            pct_gt_1 = (num_gt_1 / total_sites) * 100 if total_sites > 0 else 0.0
            
            results.append({
                'eps': float(eps_str),
                'total_sites': total_sites,
                'max_water_per_site': max_wps,
                'min_water_per_site': min_wps,
                'num_sites_gt_1': num_gt_1,
                'pct_sites_gt_1': pct_gt_1
            })
            print(f"  eps {eps_str}: total={total_sites}, max={max_wps:.6f}, min={min_wps:.6f}, "
                  f"num>1={num_gt_1} ({pct_gt_1:.2f}%)")
            
        except Exception as e:
            print(f"  Error processing {csv_path}: {e}")
            continue
    
    if results:
        out_df = pd.DataFrame(results).sort_values('eps')
        # Column order can be adjusted; here it is the default insertion order
        output_path = os.path.join(base_dir, 'max_min_water_per_site_summary.csv')
        out_df.to_csv(output_path, index=False)
        print(f"  Summary results saved to: {output_path}")
        return True
    else:
        print(f"  No valid data found under {base_dir}")
        return False

def main():
    # Parent directory (modify according to actual situation)
    parent_dir = r'D:\1-Redmi\Linux\8-BRD4-water\4-MD-hbond-network\3-watersite-complex'
    
    # Fixed eps list (only 0.30 ~ 0.90, step 0.10)
    eps_values = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    # Get all subdirectories under parent_dir
    if not os.path.exists(parent_dir):
        print(f"Error: Parent directory {parent_dir} does not exist")
        return
    
    sub_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    
    # Exclude output-2 (original script logic; can adjust exclusion list as needed)
    target_dirs = [d for d in sub_dirs if d != 'output-2']
    
    if not target_dirs:
        print("No subdirectories other than output-2 found")
        return
    
    print(f"Found the following directories to process:{target_dirs}")
    print(f"eps value list:{eps_values}\n")
    
    for dir_name in target_dirs:
        base_path = os.path.join(parent_dir, dir_name)
        print(f"Processing:{base_path}")
        process_directory(base_path, eps_values)
        print()  # Empty line separator

if __name__ == '__main__':
    main()