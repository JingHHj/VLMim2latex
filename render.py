import csv
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
import os

def safe_render_latex(latex_code):
    """Safely render LaTeX code, falling back to raw text if rendering fails"""
    if not latex_code or not latex_code.strip():
        return latex_code, 'red', False
    
    try:
        parser = mathtext.MathTextParser('path')
        wrapped_code = f"${latex_code}$"
        parser.parse(wrapped_code, dpi=72, prop=None)
        return wrapped_code, 'black', True
    except:
        return latex_code, 'red', False

def render_all_data(csv_path, output_dir='baseline_rendered_batches'):
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)
        
        for row in reader:
            idx, ground_truth, prediction, token_acc, seq_acc = row
            
            # Clean LaTeX
            gt_clean = ground_truth.replace(r'\displaystyle', '')
            pred_clean = prediction.replace(r'\displaystyle', '')
            
            all_data.append({
                'idx': idx,
                'ground_truth': gt_clean,
                'prediction': pred_clean,
                'token_accuracy': float(token_acc),
                'seq_accuracy': seq_acc
            })
    
    print(f"Found {len(all_data)} total data points")
    
    # Process in batches of 10
    batch_size = 10
    for batch_idx, start_idx in enumerate(range(0, len(all_data), batch_size)):
        batch_data = all_data[start_idx:start_idx + batch_size]
        batch_num = batch_idx + 1
        
        # Create figure with 3 columns (GT, Pred, Token Acc), rows = batch size
        fig, axes = plt.subplots(len(batch_data), 3, figsize=(5.5, 0.5*len(batch_data)))
        
        # Handle single row case
        if len(batch_data) == 1:
            axes = [axes]
        
        for i, data in enumerate(batch_data):
            # Ground Truth (first column)
            gt_text, gt_color, gt_success = safe_render_latex(data['ground_truth'])
            # gt_display = gt_text if gt_success else f"{data['ground_truth']} [FAIL COMPILED]"
            gt_display = gt_text if gt_success else f"[FAIL COMPILED]"

            
            axes[i][0].text(0.5, 0.5, gt_display, fontsize=10, ha='center', va='center', 
                           color=gt_color, wrap=True)
            axes[i][0].axis('off')
            
            # # Add row label (index)
            # axes[i][0].text(0.02, 0.98, f"#{data['idx']}", transform=axes[i][0].transAxes,
            #                fontsize=10, ha='left', va='top', fontweight='bold')
            
            # Prediction (second column)
            pred_text, pred_color, pred_success = safe_render_latex(data['prediction'])
            pred_display = pred_text if pred_success else f"{data['prediction']} [FAIL COMPILED]"
            
            axes[i][1].text(0.5, 0.5, pred_display, fontsize=10, ha='center', va='center', 
                           color=pred_color, wrap=True)
            axes[i][1].axis('off')
            
            # Token Accuracy (third column)
            token_acc_text = f"{data['token_accuracy']:.4f}"
            axes[i][2].text(0.5, 0.5, token_acc_text, fontsize=10, ha='center', va='center', 
                           color='black')
            axes[i][2].axis('off')
        
        # Add column headers only for first row
        axes[0][0].text(0.5, 1.05, 'Ground Truth', transform=axes[0][0].transAxes,
                       fontsize=12, ha='center', fontweight='bold')
        axes[0][1].text(0.5, 1.05, 'Prediction', transform=axes[0][1].transAxes,
                       fontsize=12, ha='center', fontweight='bold')
        axes[0][2].text(0.5, 1.05, 'Token Accuracy', transform=axes[0][2].transAxes,
                       fontsize=12, ha='center', fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.05, wspace=0.02)
        
        # Save batch
        output_path = os.path.join(output_dir, f'batch_{batch_num:02d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: batch_{batch_num:02d}.png ({len(batch_data)} items)")
    
    print(f"\nCreated {batch_idx + 1} batch files in '{output_dir}/' directory")

if __name__ == "__main__":
    csv_file_path = "baseline_eval_20_samples.csv" 
    
    render_all_data(csv_file_path)