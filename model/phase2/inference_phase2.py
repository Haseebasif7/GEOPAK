"""Phase 2 Inference Script - Mixture of Hypotheses"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pipeline.data_streaming.transforms import get_val_test_transforms
from model.phase2.model_utils import load_phase1_checkpoint
from model.phase1.utils import haversine_km


def mixture_of_hypotheses_inference(
    model,
    image_tensor,
    device,
    top_k_provinces=2,
    top_k_cells=5
):
    """
    Mixture of Hypotheses Inference
    
    Args:
        model: GeopakPhase1Model
        image_tensor: [1, 3, 224, 224]
        device: torch device
        top_k_provinces: Number of top provinces to consider (default: 2)
        top_k_cells: Number of top cells per province (default: 5)
    
    Returns:
        dict with prediction details
    """
    model.eval()
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor, return_all=True)
        
        # 1. Get province probabilities
        province_logits = outputs['province_logits']  # [1, 7]
        province_probs = F.softmax(province_logits, dim=1)[0]  # [7]
        
        # 2. Select Top-K provinces
        top_province_probs, top_province_ids = torch.topk(province_probs, top_k_provinces)
        
        # Storage for hypotheses
        hypotheses = []
        
        # 3. For each top province
        for prov_idx in range(top_k_provinces):
            prov_id = top_province_ids[prov_idx].item()
            prov_prob = top_province_probs[prov_idx].item()
            prov_name = model.province_names[prov_id]
            
            # Get geocell logits for this province
            if prov_id in outputs['geocell_logits']:
                geocell_info = outputs['geocell_logits'][prov_id]
                local_logits = geocell_info['local_logits'][0]  # [num_cells_in_province]
                
                # Get cell probabilities
                cell_probs = F.softmax(local_logits, dim=0)
                
                # 4. Select Top-K cells
                top_cell_probs, top_cell_local_ids = torch.topk(cell_probs, min(top_k_cells, len(cell_probs)))
                
                # Get global cell IDs
                buffer_name = model.local_to_global_cell_map[prov_id]
                local_to_global = getattr(model, buffer_name)
                
                # For each top cell
                for cell_idx in range(len(top_cell_local_ids)):
                    local_id = top_cell_local_ids[cell_idx].item()
                    global_cell_id = local_to_global[local_id].item()
                    cell_prob = top_cell_probs[cell_idx].item()
                    
                    # Get cell center
                    cell_center_lat = model.cell_centers_lat[global_cell_id].item()
                    cell_center_lon = model.cell_centers_lon[global_cell_id].item()
                    
                    # Get offset prediction (need to run offset head)
                    cell_id_tensor = torch.tensor([global_cell_id], device=device)
                    prov_id_tensor = torch.tensor([prov_id], device=device)
                    
                    cell_embed = model.cell_embedding(cell_id_tensor)
                    prov_embed = model.province_embedding(prov_id_tensor)
                    e_img = outputs['e_img']
                    
                    offset = model.offset_head(e_img, cell_embed, prov_embed)[0]  # [2]
                    offset_lat = offset[0].item()
                    offset_lon = offset[1].item()
                    
                    # 5. Compute prediction for this hypothesis
                    pred_lat = cell_center_lat + offset_lat
                    pred_lon = cell_center_lon + offset_lon
                    
                    # Combined probability: p_i = P(province) × P(cell|province)
                    combined_prob = prov_prob * cell_prob
                    
                    hypotheses.append({
                        'province_id': prov_id,
                        'province_name': prov_name,
                        'province_prob': prov_prob,
                        'cell_id': global_cell_id,
                        'cell_prob': cell_prob,
                        'combined_prob': combined_prob,
                        'cell_center_lat': cell_center_lat,
                        'cell_center_lon': cell_center_lon,
                        'offset_lat': offset_lat,
                        'offset_lon': offset_lon,
                        'pred_lat': pred_lat,
                        'pred_lon': pred_lon,
                    })
        
        # 6. Final prediction: weighted average
        total_prob = sum(h['combined_prob'] for h in hypotheses)
        
        if total_prob > 0:
            final_lat = sum(h['combined_prob'] * h['pred_lat'] for h in hypotheses) / total_prob
            final_lon = sum(h['combined_prob'] * h['pred_lon'] for h in hypotheses) / total_prob
        else:
            # Fallback to top hypothesis
            final_lat = hypotheses[0]['pred_lat']
            final_lon = hypotheses[0]['pred_lon']
        
        return {
            'final_lat': final_lat,
            'final_lon': final_lon,
            'hypotheses': hypotheses,
            'province_probs': province_probs.cpu().numpy(),
            'top_province_ids': top_province_ids.cpu().numpy(),
            'top_province_probs': top_province_probs.cpu().numpy(),
        }


def visualize_prediction(image_path, prediction, ground_truth, save_path):
    """
    Visualize prediction with image and hypothesis details
    
    Args:
        image_path: Path to input image
        prediction: Prediction dict from mixture_of_hypotheses_inference
        ground_truth: Dict with 'lat', 'lon', 'province'
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Image
    img = Image.open(image_path)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(f"Input Image\nGround Truth: {ground_truth['province']}", fontsize=12, fontweight='bold')
    
    # Right: Prediction details
    ax = axes[1]
    ax.axis('off')
    
    # Calculate error
    error_km = haversine_km(
        torch.tensor(ground_truth['lat']), torch.tensor(ground_truth['lon']),
        torch.tensor(prediction['final_lat']), torch.tensor(prediction['final_lon'])
    ).item()
    
    # Title
    title = f"Phase 2 Prediction - Error: {error_km:.2f} km"
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    # Final prediction
    final_text = f"Final Prediction:\n"
    final_text += f"  Lat: {prediction['final_lat']:.6f}°\n"
    final_text += f"  Lon: {prediction['final_lon']:.6f}°\n\n"
    final_text += f"Ground Truth:\n"
    final_text += f"  Lat: {ground_truth['lat']:.6f}°\n"
    final_text += f"  Lon: {ground_truth['lon']:.6f}°\n"
    
    ax.text(0.05, 0.80, final_text, ha='left', va='top', fontsize=11, 
            family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Top hypotheses
    hypotheses_text = "Top Hypotheses (Top-2 Provinces × Top-5 Cells):\n"
    hypotheses_text += "-" * 60 + "\n"
    
    for i, h in enumerate(prediction['hypotheses'][:10]):  # Show top 10
        hypotheses_text += f"{i+1}. {h['province_name']} (P={h['province_prob']:.3f})\n"
        hypotheses_text += f"   Cell {h['cell_id']} (P={h['cell_prob']:.3f})\n"
        hypotheses_text += f"   Combined P={h['combined_prob']:.4f}\n"
        hypotheses_text += f"   Pred: ({h['pred_lat']:.4f}°, {h['pred_lon']:.4f}°)\n"
        hypotheses_text += f"   Offset: ({h['offset_lat']:.4f}°, {h['offset_lon']:.4f}°)\n\n"
    
    ax.text(0.05, 0.55, hypotheses_text, ha='left', va='top', fontsize=8,
            family='monospace', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization: {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Inference")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/phase2/phase2_best.pt',
                        help='Path to Phase 2 checkpoint')
    parser.add_argument('--test-csv', type=str, default='test.csv',
                        help='Path to test.csv')
    parser.add_argument('--cell-metadata', type=str, default='pipeline/geocells/cell_metadata.csv',
                        help='Path to cell_metadata.csv')
    parser.add_argument('--scene-model', type=str, default='resnet50_places365.pth.tar',
                        help='Path to scene model')
    parser.add_argument('--output-dir', type=str, default='phase2_inference_results',
                        help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Paths
    checkpoint_path = project_root / args.checkpoint
    test_csv_path = project_root / args.test_csv
    cell_metadata_path = project_root / args.cell_metadata
    scene_model_path = project_root / args.scene_model
    output_dir = project_root / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("PHASE 2 INFERENCE - Mixture of Hypotheses")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    print(f"\n📂 Loading Phase 2 checkpoint: {checkpoint_path}")
    model, checkpoint_info = load_phase1_checkpoint(
        checkpoint_path=checkpoint_path,
        cell_metadata_path=cell_metadata_path,
        scene_model_path=scene_model_path if scene_model_path.exists() else None,
        device=device
    )
    model.eval()
    
    print(f"   Loaded checkpoint from epoch {checkpoint_info['epoch']}")
    
    # Load test data
    print(f"\n📦 Loading test data: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    
    # Get 1 sample per province
    provinces = ["Sindh", "Punjab", "Khyber Pakhtunkhwa", "ICT", 
                 "Gilgit-Baltistan", "Balochistan", "Azad Kashmir"]
    
    transform = get_val_test_transforms()
    
    print(f"\n🔍 Running inference on 1 sample per province...")
    print("=" * 70)
    
    results = []
    
    for province in provinces:
        # Get one sample from this province
        province_samples = test_df[test_df['province'] == province]
        
        if len(province_samples) == 0:
            print(f"\n⚠️  No samples found for {province}")
            continue
        
        # Take first sample
        sample = province_samples.iloc[0]
        
        image_path = Path(sample['path'])
        gt_lat = sample['latitude']
        gt_lon = sample['longitude']
        gt_province = sample['province']
        
        print(f"\n📍 {province}")
        print(f"   Image: {image_path.name}")
        print(f"   Ground Truth: ({gt_lat:.6f}°, {gt_lon:.6f}°)")
        
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
        
        # Run inference
        prediction = mixture_of_hypotheses_inference(
            model=model,
            image_tensor=image_tensor,
            device=device,
            top_k_provinces=2,
            top_k_cells=5
        )
        
        # Calculate error
        error_km = haversine_km(
            torch.tensor(gt_lat), torch.tensor(gt_lon),
            torch.tensor(prediction['final_lat']), torch.tensor(prediction['final_lon'])
        ).item()
        
        print(f"   Prediction: ({prediction['final_lat']:.6f}°, {prediction['final_lon']:.6f}°)")
        print(f"   Error: {error_km:.2f} km")
        
        # Top provinces
        print(f"   Top Provinces:")
        for i in range(len(prediction['top_province_ids'])):
            prov_id = prediction['top_province_ids'][i]
            prov_prob = prediction['top_province_probs'][i]
            prov_name = model.province_names[prov_id]
            print(f"      {i+1}. {prov_name}: {prov_prob:.4f}")
        
        # Visualize
        vis_path = output_dir / f"{province.lower().replace(' ', '_')}_prediction.png"
        visualize_prediction(
            image_path=image_path,
            prediction=prediction,
            ground_truth={'lat': gt_lat, 'lon': gt_lon, 'province': gt_province},
            save_path=vis_path
        )
        
        results.append({
            'province': province,
            'image': image_path.name,
            'gt_lat': gt_lat,
            'gt_lon': gt_lon,
            'pred_lat': prediction['final_lat'],
            'pred_lon': prediction['final_lon'],
            'error_km': error_km,
            'top_province_1': model.province_names[prediction['top_province_ids'][0]],
            'top_province_1_prob': prediction['top_province_probs'][0],
            'num_hypotheses': len(prediction['hypotheses']),
        })
    
    # Summary
    print("\nSUMMARY")
    
    results_df = pd.DataFrame(results)
    print(f"\nMean Error: {results_df['error_km'].mean():.2f} km")
    print(f"Median Error: {results_df['error_km'].median():.2f} km")
    print(f"Min Error: {results_df['error_km'].min():.2f} km")
    print(f"Max Error: {results_df['error_km'].max():.2f} km")
    
    # Save results
    results_csv = output_dir / 'inference_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n📊 Results saved to: {results_csv}")
    
    print(f"\n✅ Inference complete! Visualizations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
