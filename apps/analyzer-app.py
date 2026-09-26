import os
import numpy as np
import pandas as pd
import cv2
import gradio as gr
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.ndimage as ndimage

# ==========================================
# CONFIGURATION
# ==========================================
GT_DIR = Path("train")
PRED_DIR = Path("train_preds")

# ==========================================
# 1. METRIC FUNCTIONS
# ==========================================
def compute_si_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Calculates Scale-Invariant RMSE."""
    valid_mask = (gt > 0) & (~np.isnan(gt)) & (pred > 0) & (~np.isnan(pred))
    p_valid = pred[valid_mask]
    g_valid = gt[valid_mask]
    
    if len(g_valid) == 0: return float('inf')
    
    log_p = np.log(p_valid)
    log_g = np.log(g_valid)
    delta = log_p - log_g
    alpha = np.mean(log_g - log_p)
    
    si_rmse = np.sqrt(np.mean((delta + alpha) ** 2))
    return float(si_rmse)

def generate_error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Creates a spatial heatmap of structural errors."""
    valid_mask = (gt > 0) & (~np.isnan(gt)) & (pred > 0) & (~np.isnan(pred))
    error_map = np.zeros_like(gt)
    
    if not np.any(valid_mask): return error_map
    
    log_p = np.log(pred[valid_mask])
    log_g = np.log(gt[valid_mask])
    delta = log_p - log_g
    alpha = np.mean(log_g - log_p)
    
    error_vals = np.abs(delta + alpha)
    error_map[valid_mask] = error_vals
    return error_map

# ==========================================
# 2. DATA LOADING & PREPARATION
# ==========================================
def load_and_evaluate():
    print(f"Scanning '{GT_DIR}' and '{PRED_DIR}'...")
    results = []

    if not PRED_DIR.exists() or not GT_DIR.exists():
        print("Error: One or both directories not found. Please check paths.")
        return pd.DataFrame()

    # Explicitly look only for files starting with 'train_' and ending in '.npy'
    for pred_path in sorted(PRED_DIR.glob("train_*.npy")):
        
        # pred_path.stem gives 'train_000000'. We strip 'train_' to get '000000'
        idx_str = pred_path.stem.replace("train_", "")
        
        # Reconstruct the exact names from your 'train' folder screenshot
        gt_depth_path = GT_DIR / f"train_{idx_str}_depth.npy"
        gt_rgb_path = GT_DIR / f"train_{idx_str}_rgb.png"
        
        # Debugging step: if it misses, it will tell you exactly which file it couldn't find
        if not gt_depth_path.exists():
            print(f"Skipping {idx_str}: Could not find {gt_depth_path}")
            continue
        if not gt_rgb_path.exists():
            print(f"Skipping {idx_str}: Could not find {gt_rgb_path}")
            continue
            
        pred_depth = np.load(pred_path)
        gt_depth = np.load(gt_depth_path)
        
        error_score = compute_si_rmse(pred_depth, gt_depth)
        
        # Calculate average depth of the scene (ignoring invalid pixels)
        valid_gt = gt_depth[(gt_depth > 0) & (~np.isnan(gt_depth))]
        avg_scene_depth = np.mean(valid_gt) if len(valid_gt) > 0 else 0
        
        # Calculate scene complexity (variance in depth)
        std_scene_depth = np.std(valid_gt) if len(valid_gt) > 0 else 0
        
        # Calculate Ground Truth Sparsity (ratio of valid pixels to total pixels)
        valid_ratio = len(valid_gt) / gt_depth.size
        
        # Calculate Image Brightness (load as grayscale just for the math)
        rgb_gray = cv2.imread(str(gt_rgb_path), cv2.IMREAD_GRAYSCALE)
        avg_brightness = np.mean(rgb_gray) if rgb_gray is not None else 0
        
        # Calculate Image Contrast (Standard Deviation of pixel intensities)
        image_contrast = np.std(rgb_gray) if rgb_gray is not None else 0
        
        # Calculate Edge Density (Ratio of edge pixels to total pixels)
        if rgb_gray is not None:
            edges = cv2.Canny(rgb_gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
        else:
            edge_density = 0
            
        results.append({
            'id': idx_str,
            'avg_depth': avg_scene_depth,
            'std_depth': std_scene_depth,
            'valid_ratio': valid_ratio,
            'brightness': avg_brightness,
            'contrast': image_contrast,
            'edge_density': edge_density,
            'rgb_path': str(gt_rgb_path),
            'gt_path': str(gt_depth_path),
            'pred_path': str(pred_path),
            'siRMSE': error_score
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='siRMSE', ascending=True).reset_index(drop=True)
        print(f"Evaluated {len(df)} matching images.")
        print(f"Best Score: {df.iloc[0]['siRMSE']:.4f}")
        print(f"Worst Score: {df.iloc[-1]['siRMSE']:.4f}")
    else:
        print("Error: No matching file pairs found.")
        
    return df

# ==========================================
# 3. GRADIO UI COMPONENTS
# ==========================================
def colorize_depth(depth_map, cmap=cv2.COLORMAP_PLASMA):
    valid = depth_map[depth_map > 0]
    if len(valid) == 0: return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    
    vmin, vmax = np.percentile(valid, 2), np.percentile(valid, 98)
    norm_depth = np.clip((depth_map - vmin) / (vmax - vmin + 1e-6), 0, 1)
    
    depth_8bit = (norm_depth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_8bit, cmap)
    colored[depth_map <= 0] = [0, 0, 0] 
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

def generate_boundary_diagnostics(pred, gt, sirms_score):
    valid_mask = (gt > 0) & (~np.isnan(gt)) & (pred > 0) & (~np.isnan(pred))
    if not np.any(valid_mask): return go.Figure()

    # 1. Find depth discontinuities (gt_disc) using gradients
    dy, dx = np.gradient(gt)
    grad_mag = np.sqrt(dx**2 + dy**2)
    grad_mag[~valid_mask] = 0
    
    # Define a boundary as the top 5% of sharpest depth jumps
    jump_threshold = np.percentile(grad_mag[valid_mask], 95)
    boundaries = (grad_mag > jump_threshold) & valid_mask

    # 2. Calculate distance to the nearest boundary pixel
    # distance_transform computes distance to 0s, so we invert the boundary mask
    dist_to_boundary = ndimage.distance_transform_edt(~boundaries)

    # 3. Compute Squared Errors (the "mass" of the error)
    log_p = np.log(pred[valid_mask])
    log_g = np.log(gt[valid_mask])
    delta = log_p - log_g
    alpha = np.mean(log_g - log_p)
    sq_errors = (delta + alpha)**2

    full_sq_errors = np.zeros_like(gt)
    full_sq_errors[valid_mask] = sq_errors
    total_error_mass = np.sum(sq_errors)
    total_valid_area = np.sum(valid_mask)

    # 4. Binning logic matching your screenshot
    bins = [0, 1, 3, 7, 15, 31, float('inf')]
    bin_labels = ["[0,1)", "[1,3)", "[3,7)", "[7,15)", "[15,31)", "[31,inf)"]
    
    mass_fracs = []
    concentration_ratios = []

    for i in range(len(bins)-1):
        lower, upper = bins[i], bins[i+1]
        bin_mask = (dist_to_boundary >= lower) & (dist_to_boundary < upper) & valid_mask
        
        area_frac = np.sum(bin_mask) / total_valid_area if total_valid_area > 0 else 0
        mass_frac = np.sum(full_sq_errors[bin_mask]) / total_error_mass if total_error_mass > 0 else 0
        
        mass_fracs.append(mass_frac)
        concentration_ratios.append(mass_frac / area_frac if area_frac > 0 else 0)

    # 5. Build the dual Plotly Figure
    # Color code the charts based on the global siRMSE score severity
    if sirms_score < 0.2:
        chart_color = "#2ca02c" # Green (Excellent)
    elif sirms_score < 0.5:
        chart_color = "#1f77b4" # Blue (Good)
    elif sirms_score < 1.0:
        chart_color = "#ff7f0e" # Orange (Poor)
    else:
        chart_color = "#d62728" # Red (Terrible)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Where does the error live?", "Boundary-error concentration ratio"))
    
    # Left: Bar Chart
    fig.add_trace(go.Bar(x=bin_labels, y=mass_fracs, name="gt_disc", marker_color=chart_color), row=1, col=1)
    
    # Right: Line Chart
    fig.add_trace(go.Scatter(x=bin_labels, y=concentration_ratios, mode='lines+markers', name="gt_disc", marker_color=chart_color), row=1, col=2)
    
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=1, col=2, annotation_text="uniform (=1)")

    fig.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_yaxes(title_text="fraction of total siRMSE^2 mass", row=1, col=1)
    fig.update_yaxes(title_text="error_mass_frac / area_frac", row=1, col=2)
    fig.update_xaxes(title_text="distance to nearest boundary (px)", row=1, col=1)
    fig.update_xaxes(title_text="distance to nearest boundary (px)", row=1, col=2)

    return fig

def create_app(df_results):
    def view_case(view_type, rank_idx):
        if df_results.empty: return None, None, None, None, None, "No data to display."
        
        # Ensure slider doesn't go out of bounds
        rank_idx = min(rank_idx, len(df_results))
        
        if view_type == "Best (Lowest Error)":
            row = df_results.iloc[rank_idx - 1]
            rank_text = f"Best Rank: #{rank_idx}"
        else:
            row = df_results.iloc[-rank_idx]
            rank_text = f"Worst Rank: #{rank_idx}"
            
        rgb_img = cv2.cvtColor(cv2.imread(row['rgb_path']), cv2.COLOR_BGR2RGB)
        gt_depth = np.load(row['gt_path'])
        pred_depth = np.load(row['pred_path'])
        
        err_map = generate_error_map(pred_depth, gt_depth)
        
        vis_gt = colorize_depth(gt_depth, cv2.COLORMAP_PLASMA)
        vis_pred = colorize_depth(pred_depth, cv2.COLORMAP_PLASMA)
        valid_gt_mask = (gt_depth > 0) & (~np.isnan(gt_depth))
        pred_depth_masked = pred_depth.copy()
        pred_depth_masked[~valid_gt_mask] = 0 # Invalid regions set to 0 (rendered black by colorize_depth)
        vis_pred_masked = colorize_depth(pred_depth_masked, cv2.COLORMAP_PLASMA)
        
        vis_err = colorize_depth(err_map, cv2.COLORMAP_JET) 
        
        # Generate Error Distribution Histogram
        valid_errs = err_map[err_map > 0]
        if len(valid_errs) > 0:
            fig_hist = px.histogram(x=valid_errs, nbins=50, 
                                    title="Pixel Error Distribution (Current Image)",
                                    labels={'x': 'Absolute Log Error', 'count': 'Number of Pixels'},
                                    color_discrete_sequence=['indianred'],
                                    log_y=True)

            fig_hist.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=450, 
                width=450,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(240,244,255,0.5)"
            )
        else:
            fig_hist = go.Figure()
        
        # Generate new Boundary plots
        fig_boundary = generate_boundary_diagnostics(pred_depth, gt_depth, row['siRMSE'])
        
        info_text = f"**Image ID:** {row['id']} | **{rank_text}** | **siRMSE Score:** {row['siRMSE']:.4f}"
        return rgb_img, vis_gt, vis_pred, vis_pred_masked, vis_err, fig_hist, fig_boundary, info_text
    
    with gr.Blocks(title="Depth Estimation Dashboard") as app:
        gr.Markdown("## 🔍 Depth Estimation Analyzer")
        
        if df_results.empty:
            gr.Markdown("### ⚠️ Error: No data found. Please check your folder paths and file names.")
            return app

        with gr.Tabs():
            with gr.TabItem("🔍 Single Image Inspector"):
                # ROW 1: Sorting and Searching
                with gr.Row(equal_height=True):
                    view_type = gr.Radio(["Worst (Highest Error)", "Best (Lowest Error)"], value="Worst (Highest Error)", label="Sort By", scale=1)
                    search_box = gr.Textbox(label="Search Image ID", placeholder="Enter ID (e.g., 009574) and press Enter...", scale=1)
                
                # ROW 2: Pagination / Slider
                with gr.Row(equal_height=True):
                    btn_prev = gr.Button("⬅️ Previous", size="lg", variant="secondary")
                    rank_idx = gr.Slider(minimum=1, maximum=len(df_results), step=1, value=1, label="Rank Index", scale=4)
                    btn_next = gr.Button("Next ➡️", size="lg", variant="secondary")
                    
                info_display = gr.Markdown("### Loading...")                
                with gr.Row():
                    img_rgb = gr.Image(label="Original Image")
                    img_pred = gr.Image(label="Predicted Depth")
                    plot_hist = gr.Plot(label="Error Distribution")
                with gr.Row():
                    img_err = gr.Image(label="Error Heatmap (Red = High Error)")
                    img_gt = gr.Image(label="Ground Truth Depth")
                    img_pred_masked = gr.Image(label="Predicted Depth (Masked)")

                with gr.Row():
                    plot_boundary = gr.Plot(label="Residual Diagnostics")
                    
                inputs = [view_type, rank_idx]
                outputs = [img_rgb, img_gt, img_pred, img_pred_masked, img_err, plot_hist, plot_boundary, info_display]
                
                # 1. Main Image Loading Callbacks
                view_type.change(fn=view_case, inputs=inputs, outputs=outputs)
                rank_idx.change(fn=view_case, inputs=inputs, outputs=outputs)
                app.load(fn=view_case, inputs=inputs, outputs=outputs)
                
                # 2. Search Callback (Finds the ID, calculates its rank, and moves the slider)
                def handle_search(search_id, current_view, current_rank):
                    if search_id in df_results['id'].values:
                        row_idx = df_results.index[df_results['id'] == search_id].tolist()[0]
                        # Calculate what rank this row is based on the current sorting view
                        if current_view == "Best (Lowest Error)":
                            new_rank = row_idx + 1
                        else:
                            new_rank = len(df_results) - row_idx
                        return new_rank, "" # Move slider to new rank, clear search box
                    return current_rank, "⚠️ ID Not Found" # Keep current rank, show error
                    
                search_box.submit(fn=handle_search, 
                                  inputs=[search_box, view_type, rank_idx], 
                                  outputs=[rank_idx, search_box])
                
                # 3. Next/Prev Button Callbacks (Moves the slider +/- 1 safely)
                btn_prev.click(fn=lambda r: max(r - 1, 1), inputs=rank_idx, outputs=rank_idx)
                btn_next.click(fn=lambda r: min(r + 1, len(df_results)), inputs=rank_idx, outputs=rank_idx)

            with gr.TabItem("🌍 Global Dataset Analysis"):
                gr.Markdown("### 📊 Global Performance Dashboard")
                gr.Markdown("Use these plots to identify systemic weaknesses in your model across the entire dataset.")
                
                # Prepare distance categories for the box plot
                if not df_results.empty:
                    df_results['Distance Category'] = pd.qcut(df_results['avg_depth'], q=4, labels=['Close', 'Medium-Close', 'Medium-Far', 'Far'], duplicates='drop')

                with gr.Row():
                    # Plot 1: Error vs Distance (Colored by Severity)
                    fig_scatter_dist = px.scatter(df_results, x='avg_depth', y='siRMSE', 
                                             color='siRMSE', color_continuous_scale='Turbo',
                                             hover_data=['id'], trendline="ols",
                                             labels={'avg_depth': 'Average Scene Depth', 'siRMSE': 'siRMSE Score'},
                                             title="Does Distance Increase Error?")
                    gr.Plot(value=fig_scatter_dist)

                    # Plot 2: Error vs Scene Complexity (Colored by Severity)
                    fig_scatter_comp = px.scatter(df_results, x='std_depth', y='siRMSE', 
                                             color='siRMSE', color_continuous_scale='Turbo',
                                             hover_data=['id'], trendline="ols",
                                             labels={'std_depth': 'Scene Complexity (Depth Std Dev)', 'siRMSE': 'siRMSE Score'},
                                             title="Do Complex Geometries Increase Error?")
                    gr.Plot(value=fig_scatter_comp)

                with gr.Row():
                    # Plot 3: Global Error Histogram
                    fig_global_hist = px.histogram(df_results, x='siRMSE', nbins=100,
                                                   color_discrete_sequence=['#636EFA'],
                                                   title="Overall siRMSE Distribution (All Images)",
                                                   labels={'siRMSE': 'siRMSE Score'})
                    gr.Plot(value=fig_global_hist)
                    
                    # Plot 4: Box Plot of Errors Grouped by Distance
                    if not df_results.empty:
                        fig_box = px.box(df_results, x='Distance Category', y='siRMSE',
                                         color='Distance Category',
                                         title="Error Spread by Scene Distance Range",
                                         labels={'siRMSE': 'siRMSE Score', 'Distance Category': 'Physical Distance Tier'})
                        gr.Plot(value=fig_box)

                with gr.Row():
                    # Plot 5: 2D Density Heatmap (Solves Scatter Overplotting)
                    fig_density = px.density_heatmap(df_results, x='avg_depth', y='siRMSE', 
                                                     nbinsx=50, nbinsy=50, color_continuous_scale="Viridis",
                                                     title="Error Density (Where does the bulk of data lie?)",
                                                     labels={'avg_depth': 'Average Scene Depth', 'siRMSE': 'siRMSE Score'})
                    gr.Plot(value=fig_density)
                    
                    # Plot 6: Lighting Sensitivity
                    fig_lighting = px.scatter(df_results, x='brightness', y='siRMSE', 
                                              color='siRMSE', color_continuous_scale='Turbo',
                                              hover_data=['id'], trendline="lowess",
                                              title="Lighting Sensitivity (Does darkness break the model?)",
                                              labels={'brightness': 'Average Pixel Brightness (0-255)', 'siRMSE': 'siRMSE Score'})
                    gr.Plot(value=fig_lighting)

                    # Plot 7: Data Sparsity Artifacts
                    fig_sparsity = px.scatter(df_results, x='valid_ratio', y='siRMSE',
                                              color='siRMSE', color_continuous_scale='Turbo',
                                              title="Is High Error caused by Sparse Ground Truth?",
                                              labels={'valid_ratio': 'Fraction of Valid GT Pixels', 'siRMSE': 'siRMSE Score'})
                    gr.Plot(value=fig_sparsity)
                    
                with gr.Row():
                    # Plot 8: Contrast vs Error
                    fig_contrast = px.scatter(df_results, x='contrast', y='siRMSE', 
                                              color='siRMSE', color_continuous_scale='Turbo',
                                              hover_data=['id'], trendline="lowess",
                                              title="Contrast Sensitivity (Dynamic Range)",
                                              labels={'contrast': 'Image Contrast (Std Dev of Grayscale)', 'siRMSE': 'siRMSE Score'})
                    gr.Plot(value=fig_contrast)
                    
                    # Plot 9: Edge Density vs Error
                    fig_edges = px.scatter(df_results, x='edge_density', y='siRMSE', 
                                              color='siRMSE', color_continuous_scale='Turbo',
                                              hover_data=['id'], trendline="lowess",
                                              title="Clutter/Texture (Edge Density)",
                                              labels={'edge_density': 'Fraction of Edge Pixels', 'siRMSE': 'siRMSE Score'})
                    gr.Plot(value=fig_edges)
        
    return app

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df_eval = load_and_evaluate()
    dashboard = create_app(df_eval)
    
    dashboard.launch(inbrowser=True, server_name="127.0.0.1", theme=gr.themes.Soft())