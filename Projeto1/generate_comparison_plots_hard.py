"""
Script to generate comparison plots for hard images showing original images with v3 and v26 predictions.
Each plot shows: Original | V3 Prediction | V26 Prediction
"""

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def create_three_way_comparison(original_path, v3_pred_path, v26_pred_path, output_path, image_name):
    """
    Create a three-panel comparison plot.
    
    Args:
        original_path: Path to the original image
        v3_pred_path: Path to the v3 prediction
        v26_pred_path: Path to the v26 prediction
        output_path: Path to save the comparison plot
        image_name: Name of the image for the title
    """
    # Load images
    original = Image.open(original_path)
    v3_pred = Image.open(v3_pred_path)
    v26_pred = Image.open(v26_pred_path)
    
    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Plot original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Plot v3 prediction
    axes[1].imshow(v3_pred)
    if "cartoon_dog" in image_name:
        axes[1].set_title('V3 Prediction', fontsize=16, fontweight='bold', y=0.97)
        axes[1].axis('off')
    else:
        axes[1].set_title('V3 Prediction', fontsize=16, fontweight='bold')
        axes[1].axis('off')
    
    # Plot v26 prediction
    axes[2].imshow(v26_pred)
    axes[2].set_title('V26 Prediction', fontsize=16, fontweight='bold')
    axes[2].axis('off')
    
    # Add overall title - remove file extension for cleaner display
    display_name = image_name.rsplit('.', 1)[0]  # Remove extension
    fig.suptitle(f'Comparison (Hard): {display_name}', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save the comparison plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def generate_all_comparisons_hard(base_dir="exps", output_dir="comparisons_hard"):
    """
    Generate three-way comparison plots for all hard images.
    
    Args:
        base_dir: Base directory containing experiments
        output_dir: Directory to save comparison plots
    """
    v3_path = Path(base_dir) / "exp_v3_hard"
    image_path = Path("images/hard_images")
    v26_path = Path(base_dir) / "exp_v26_hard"
    
    if not v3_path.exists() or not v26_path.exists():
        print("⚠️  Experiment directories not found")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # List of hard test images
    image_files = [
        'cartoon_dog.png',
        'city_scene_cartoon.png',
        'crowded_crossway.png',
        'dog_remix_cartoon.png',
        'flock_birds.png'
    ]
    
    print("\n" + "="*60)
    print("GENERATING THREE-WAY COMPARISON PLOTS (HARD IMAGES)")
    print("Original | V3 Prediction | V26 Prediction")
    print("="*60 + "\n")
    
    for img_file in image_files:
        v3_pred = v3_path / img_file
        v26_pred = v26_path / img_file
        original_img = image_path / img_file
        
        if not v3_pred.exists() or not v26_pred.exists():
            print(f"⚠️  Skipping {img_file} (prediction files not found)")
            continue
        
        if not original_img.exists():
            print(f"⚠️  Skipping {img_file} (original file not found)")
            continue
        
        output_file = output_path / f"comparison_{img_file}"
        
        create_three_way_comparison(
            original_img,
            v3_pred,
            v26_pred,
            output_file,
            img_file
        )
    
    print("\n" + "="*60)
    print("✅ ALL COMPARISON PLOTS GENERATED (HARD IMAGES)")
    print(f"Output directory: {output_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    generate_all_comparisons_hard()

   

# Made with Bob
