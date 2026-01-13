import os
import wandb
from wandb.apis.public.files import Files

# -----------------------------
# Setup
# -----------------------------
api = wandb.Api()

run_id = "cube_stacking_gym_rlpd_20260112_201532"
project_path = f"/nannuriabhi2000-hochschule-schmalkalden/hil-serl/{run_id}"

run = api.run(project_path)

# Base output directory
base_dir = f"/home/qte9489/personal_abhi/temp/hil-serl/examples/experiments/cube_stacking_gym/RLPD_Checkpoints/trial2/{run_id}"
os.makedirs(base_dir, exist_ok=True)

print(f"Downloading W&B data for run: {run_id}")
print(f"Saving to: {base_dir}")

# -----------------------------
# 1. Download full history (all metrics)
# -----------------------------
print("\n[1] Downloading full history (metrics, histograms, distributions)...")

history_df = run.history(samples=None)  # full history
csv_path = os.path.join(base_dir, "history_full.csv")
history_df.to_csv(csv_path, index=False)

print(f"Saved history to: {csv_path}")
print(f"History columns: {list(history_df.columns)}")

# -----------------------------
# 2. Download summary metrics
# -----------------------------
import json
from wandb.old.summary import SummarySubDict

def make_json_safe(obj):
    """Recursively convert W&B Summary objects into JSON-safe types."""
    
    # Case 1: W&B SummarySubDict → convert to normal dict
    if isinstance(obj, SummarySubDict):
        return {k: make_json_safe(v) for k, v in obj._dict.items()}
    
    # Case 2: Normal dict
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    
    # Case 3: List or tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    
    # Case 4: Primitive JSON-safe types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    
    # Case 5: Anything else → convert to string
    return str(obj)

print("\n[2] Downloading summary metrics...")
summary_path = os.path.join(base_dir, "summary.json")

safe_summary = make_json_safe(dict(run.summary))

with open(summary_path, "w") as f:
    json.dump(safe_summary, f, indent=4)

print(f"Saved summary to: {summary_path}")


# -----------------------------
# 3. Download all files (images, plots, artifacts, media)
# -----------------------------
print("\n[3] Downloading all files from W&B run...")

files = Files(api.client, run)

for file in files:
    print(f"Downloading: {file.name}  ({file.size} bytes)")
    file.download(root=base_dir, replace=True)

print("All files downloaded.")

# -----------------------------
# 4. Download images separately into /images folder
# -----------------------------
print("\n[4] Extracting and downloading image files...")

images_dir = os.path.join(base_dir, "images")
os.makedirs(images_dir, exist_ok=True)

for f in run.files():
    if f.name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
        print(f"Image: {f.name}")
        f.download(root=images_dir, replace=True)

print("All images downloaded.")

# -----------------------------
# 5. Download media (videos, gifs, etc.)
# -----------------------------
print("\n[5] Extracting and downloading media files...")

media_dir = os.path.join(base_dir, "media")
os.makedirs(media_dir, exist_ok=True)

for f in run.files():
    if f.name.lower().endswith((".mp4", ".webm", ".gif")):
        print(f"Media: {f.name}")
        f.download(root=media_dir, replace=True)

print("All media downloaded.")

print("\nDone! All W&B data has been saved locally.")

# -----------------------------
# 6. Generate plots from history data
# -----------------------------
print("\n[6] Generating plots from history data...")

import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import ast

# Read the history CSV
csv_path = os.path.join(base_dir, "history_full.csv")
df = pd.read_csv(csv_path)

print(f"Available columns: {list(df.columns)}")

# Filter columns that contain actual metric data (exclude system columns)
metric_columns = [col for col in df.columns if not col.startswith('_') and col not in ['_step', '_timestamp', '_runtime']]

print(f"\nGenerating plots for {len(metric_columns)} metrics...")

plots_dir = os.path.join(base_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Generate individual plots for each metric
for metric in metric_columns:
    if df[metric].notna().any():  # Only plot if there's non-null data
        try:
            # Check if this is histogram data
            first_valid = df[metric].dropna().iloc[0]
            is_histogram = False
            
            if isinstance(first_valid, str):
                try:
                    parsed = ast.literal_eval(first_valid)
                    if isinstance(parsed, dict) and parsed.get('_type') == 'histogram':
                        is_histogram = True
                except:
                    pass
            
            if is_histogram:
                # Generate histogram visualization
                print(f"  Generating histogram plot for: {metric}")
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                # Process histogram data for each step
                steps = []
                all_bins = []
                all_values = []
                
                for idx, row in df[['_step', metric]].dropna().iterrows():
                    try:
                        hist_data = ast.literal_eval(row[metric])
                        if hist_data.get('_type') == 'histogram':
                            values = hist_data.get('values', [])
                            bins_data = hist_data.get('packedBins', {})
                            
                            # Reconstruct bin edges from packed bins format
                            min_val = bins_data.get('min', 0)
                            bin_size = bins_data.get('size', 1)
                            n_bins = bins_data.get('count', len(values))
                            
                            if n_bins > 0:
                                # Create bin edges: min, min+size, min+2*size, ..., min+n*size
                                bin_edges = [min_val + i * bin_size for i in range(n_bins + 1)]
                                bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
                                
                                steps.append(row['_step'])
                                all_bins.append(bin_centers)
                                all_values.append(values)
                    except:
                        continue
                
                if steps and all_bins and all_values:
                    # Create 2D histogram plot (heatmap style)
                    # Flatten data for visualization
                    max_bins = max(len(b) for b in all_bins)
                    
                    # Create grid for imshow
                    grid = np.zeros((max_bins, len(steps)))
                    bin_positions = np.zeros((max_bins, len(steps)))
                    
                    for step_idx, (step, bins, values) in enumerate(zip(steps, all_bins, all_values)):
                        for bin_idx, (bin_center, value) in enumerate(zip(bins, values)):
                            if bin_idx < max_bins:
                                grid[bin_idx, step_idx] = value
                                bin_positions[bin_idx, step_idx] = bin_center
                    
                    # Use the first step's bin centers for y-axis
                    y_labels = all_bins[0] if all_bins[0] is not None else np.arange(max_bins)
                    
                    # Create heatmap
                    im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
                    
                    # Set axis labels
                    ax.set_xlabel('Training Step', fontsize=12)
                    ax.set_ylabel('Action Value', fontsize=12)
                    ax.set_title(f'{metric} Distribution over Training', fontsize=14, fontweight='bold')
                    
                    # Set x-axis ticks to show actual steps
                    step_ticks = np.linspace(0, len(steps)-1, min(10, len(steps)), dtype=int)
                    ax.set_xticks(step_ticks)
                    ax.set_xticklabels([f'{steps[i]:.0f}' for i in step_ticks])
                    
                    # Set y-axis ticks to show actual bin values
                    y_ticks = np.linspace(0, len(y_labels)-1, min(10, len(y_labels)), dtype=int)
                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels([f'{y_labels[i]:.2f}' for i in y_ticks])
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Frequency', fontsize=11)
                    
                    # Save plot
                    plot_filename = metric.replace('/', '_').replace(' ', '_') + '_histogram.png'
                    plot_path = os.path.join(plots_dir, plot_filename)
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ Saved histogram: {plot_filename}")
                else:
                    print(f"  ✗ No valid histogram data for {metric}")
            else:
                # Regular line plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot the metric
                valid_data = df[['_step', metric]].dropna()
                if len(valid_data) > 0:
                    ax.plot(valid_data['_step'], valid_data[metric], linewidth=2)
                    ax.set_xlabel('Step', fontsize=12)
                    ax.set_ylabel(metric, fontsize=12)
                    ax.set_title(f'{metric} over Training', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Save plot
                    plot_filename = metric.replace('/', '_').replace(' ', '_') + '.png'
                    plot_path = os.path.join(plots_dir, plot_filename)
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ Saved: {plot_filename}")
        except Exception as e:
            print(f"  ✗ Failed to plot {metric}: {e}")

# Generate combined plots for related metrics
print("\nGenerating combined plots...")

# BC loss plot
bc_metrics = [col for col in metric_columns if 'bc' in col.lower() and 'loss' in col.lower()]
if bc_metrics:
    fig, ax = plt.subplots(figsize=(12, 7))
    for metric in bc_metrics:
        valid_data = df[['_step', metric]].dropna()
        if len(valid_data) > 0:
            ax.plot(valid_data['_step'], valid_data[metric], label=metric, linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('BC Training Losses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'bc_losses_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: bc_losses_combined.png")

# Eval metrics plot
eval_metrics = [col for col in metric_columns if 'eval' in col.lower()]
if eval_metrics:
    fig, axes = plt.subplots(len(eval_metrics), 1, figsize=(12, 5*len(eval_metrics)))
    if len(eval_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(eval_metrics):
        valid_data = df[['_step', metric]].dropna()
        if len(valid_data) > 0:
            axes[idx].plot(valid_data['_step'], valid_data[metric], linewidth=2, color='green')
            axes[idx].set_xlabel('Step', fontsize=12)
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].set_title(metric, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'eval_metrics_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: eval_metrics_combined.png")

print(f"\nAll plots saved to: {plots_dir}")
print("\n✅ Export complete! Check the following directories:")
print(f"   - CSV data: {csv_path}")
print(f"   - Plots: {plots_dir}")
print(f"   - Config: {os.path.join(base_dir, 'config.yaml')}")
print(f"   - Summary: {os.path.join(base_dir, 'summary.json')}")
