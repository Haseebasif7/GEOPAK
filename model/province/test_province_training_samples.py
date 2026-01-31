"""
Test province model with 1 test image per province and display results.

Usage:
    python model/province/test_province_training_samples.py --checkpoint checkpoints/province/province_best.pt
"""

import sys
import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.province.province_head import ProvinceHead
from pipeline.data_streaming.transforms import get_val_test_transforms


def load_province_mapping():
    """Load province name to ID mapping"""
    mapping_path = project_root / "model" / "province_mapping.json"
    with open(mapping_path, 'r') as f:
        return json.load(f)


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get temperature from checkpoint if available, otherwise use default
    temperature = checkpoint.get('temperature', 1.0)
    
    # Try to find scene model path
    scene_model_path = None
    for path in [
        project_root / "resnet50_places365.pth.tar",
        checkpoint_path.parent.parent.parent / "resnet50_places365.pth.tar",
    ]:
        if path.exists():
            scene_model_path = path
            break
    
    # Initialize model with same architecture as training
    model = ProvinceHead(
        fusion_dim=512,
        hidden_dim=256,
        freeze_clip=True,
        freeze_scene=True,
        scene_model_path=str(scene_model_path) if scene_model_path else None,
        temperature=temperature
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   ✅ Model loaded successfully")
    print(f"   📊 Checkpoint info:")
    print(f"      Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"      Accuracy: {checkpoint.get('accuracy', 'unknown'):.2f}%")
    print(f"      Temperature: {temperature}")
    
    return model


def predict_province(model, image_path, device, province_mapping):
    """Predict province for a single image"""
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # Apply validation transforms (resize + to tensor)
    transform = get_val_test_transforms()
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Run inference
    with torch.no_grad():
        logits = model(image_tensor)  # [1, 7]
        probabilities = F.softmax(logits, dim=1)  # [1, 7]
        predicted_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_id].item()
    
    # Map ID to province name
    id_to_province = {v: k for k, v in province_mapping.items()}
    predicted_province = id_to_province.get(predicted_id, "Unknown")
    
    # Get all probabilities
    all_probs = {}
    for province, prov_id in province_mapping.items():
        all_probs[province] = probabilities[0][prov_id].item()
    
    return predicted_province, confidence, all_probs, image


def select_samples_per_province(test_csv_path, num_per_province=1):
    """Select num_per_province samples from each province (random each run)"""
    df = pd.read_csv(test_csv_path)
    
    # Group by province and sample (no random_state for true randomness)
    samples = []
    for province in df['province'].unique():
        province_df = df[df['province'] == province]
        sampled = province_df.sample(n=min(num_per_province, len(province_df)))
        samples.append(sampled)
    
    result_df = pd.concat(samples, ignore_index=True)
    return result_df


def display_results(results, province_mapping):
    """Display images with predictions in a grid"""
    num_provinces = len(results)
    cols = 3
    rows = (num_provinces + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # Province colors for visualization
    province_colors = {
        'Sindh': '#FF6B6B',
        'Punjab': '#4ECDC4',
        'Khyber Pakhtunkhwa': '#45B7D1',
        'ICT': '#FFA07A',
        'Gilgit-Baltistan': '#98D8C8',
        'Balochistan': '#F7DC6F',
        'Azad Kashmir': '#BB8FCE'
    }
    
    for idx, (province, data) in enumerate(results.items()):
        ax = axes[idx]
        
        # Display image
        image = data['image']
        ax.imshow(image)
        ax.axis('off')
        
        # Get prediction info
        true_province = data['true_province']
        pred_province = data['predicted_province']
        confidence = data['confidence']
        all_probs = data['all_probs']
        
        # Determine if prediction is correct
        is_correct = (true_province == pred_province)
        color = 'green' if is_correct else 'red'
        
        # Title with prediction
        title = f"True: {true_province}\n"
        title += f"Predicted: {pred_province} ({confidence*100:.1f}%)\n"
        title += f"{'✅ CORRECT' if is_correct else '❌ WRONG'}"
        ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=10)
        
        # Add border color based on correctness
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        
        # Add probability bar chart below image
        # Get top 3 predictions
        top_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        prob_text = "Top 3: " + ", ".join([f"{p[0]}: {p[1]*100:.1f}%" for p in top_probs])
        ax.text(0.5, -0.05, prob_text, transform=ax.transAxes, 
                ha='center', fontsize=9, style='italic')
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test Province Model on Test Samples')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/province/province_best.pt',
                       help='Path to checkpoint file')
    parser.add_argument('--test_csv', type=str,
                       default='test.csv',
                       help='Path to test.csv file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'mps', 'cuda'],
                       help='Device to use (default: auto)')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save the visualization (optional)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 80)
    print("PROVINCE MODEL TEST - TEST SAMPLES")
    print("=" * 80)
    print(f"\n🖥️  Device: {device}")
    
    # Load province mapping
    province_mapping = load_province_mapping()
    print(f"\n📋 Loaded province mapping: {len(province_mapping)} provinces")
    
    # Load model
    checkpoint_path = project_root / args.checkpoint
    model = load_model(checkpoint_path, device)
    
    # Load test data and select samples
    test_csv_path = project_root / args.test_csv
    if not test_csv_path.exists():
        print(f"\n❌ Test CSV not found: {test_csv_path}")
        sys.exit(1)
    
    print(f"\n📊 Loading test samples from: {test_csv_path}")
    samples_df = select_samples_per_province(test_csv_path, num_per_province=1)
    print(f"   Selected {len(samples_df)} samples (1 per province)")
    print()
    
    # Run predictions
    print("=" * 80)
    print("RUNNING PREDICTIONS")
    print("=" * 80)
    print()
    
    results = {}
    correct = 0
    total = 0
    
    for idx, row in samples_df.iterrows():
        province = row['province']
        image_path = Path(row['path'])
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        print(f"Testing {province}...")
        print(f"  Image: {image_path.name}")
        
        try:
            predicted_province, confidence, all_probs, image = predict_province(
                model, image_path, device, province_mapping
            )
            
            is_correct = (province == predicted_province)
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"  {status} Predicted: {predicted_province} ({confidence*100:.2f}%)")
            print(f"     Top 3: {sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
            results[province] = {
                'true_province': province,
                'predicted_province': predicted_province,
                'confidence': confidence,
                'all_probs': all_probs,
                'image': image,
                'image_path': str(image_path)
            }
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    # Display results
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print()
    
    # Create visualization
    print("Creating visualization...")
    fig = display_results(results, province_mapping)
    
    # Save or show
    if args.save_path:
        save_path = project_root / args.save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {save_path}")
    else:
        plt.show()
    
    print("\n" + "=" * 80)
    print("✅ Test complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
