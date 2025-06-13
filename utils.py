from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import wandb
from datasets import load_dataset
import os
import datasets
from itertools import islice
from torchvision import transforms


class Im2LatexDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        text = "<OCR>" 
        answer = example['text']
        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        return text, answer, image
    

def run_example(image, processor, model, device):
    task_prompt = "<OCR>"
    prompt = task_prompt 

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.lora_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer





def safe_render_latex(latex_code):
    """Safely render LaTeX code, falling back to raw text if rendering fails"""
    if not latex_code or not latex_code.strip():
        return "Empty LaTeX", 'red', False
    
    try:
        # Test if matplotlib can parse the LaTeX
        import matplotlib.mathtext as mathtext
        parser = mathtext.MathTextParser('path')
        wrapped_code = f"${latex_code}$"
        
        # Try to parse without actually rendering
        parser.parse(wrapped_code, dpi=72, prop=None)
        
        return wrapped_code, 'black', True  # latex_text, color, success
    except Exception as e:
        print(f"LaTeX parsing failed for '{latex_code[:50]}...': {str(e)[:100]}...")
        # Return original code as plain text (no $ wrapping)
        return latex_code, 'red', False

def render(latex_codes, image_tensor, image_sizes, output_path=None, log_wandb=False, dpi=300):
    B = image_tensor.shape[0]
    
    # Create figure with better size management
    fig_width = 12
    fig_height = max(2, 2 * B)  # Ensure minimum height
    fig, axs = plt.subplots(B, 2, figsize=(fig_width, fig_height))
    
    if B == 1:
        axs = [axs]  # Make it iterable
    
    subplot_height_inch = fig_height / B
    subplot_height_px = int(subplot_height_inch * dpi)
    
    for i in range(B):
        # Extract original LaTeX code (preserve as-is)
        latex = latex_codes[i]['<OCR>'].replace(r'\displaystyle', '')
        
        # Try to render LaTeX safely, fall back to raw text if it fails
        display_text, text_color, render_success = safe_render_latex(latex)
        
        if not render_success:
            print(f"Warning: LaTeX rendering failed at index {i}")
        
        # Render LaTeX/text
        axs[i][0].text(0.5, 0.5, display_text, fontsize=14, ha='center', va='center', 
                       color=text_color, wrap=True)
        axs[i][0].axis('off')
        axs[i][0].set_title("LaTeX Output", fontsize=12)
        
        # Prepare and display image
        try:
            img = image_tensor[i].detach().cpu().permute(1, 2, 0).numpy()
            
            # Handle different image formats (grayscale vs RGB)
            if img.shape[2] == 1:
                img = img.squeeze(2)
                cmap = 'gray'
            else:
                cmap = None
            
            # Normalize image to 0-1 range
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            
            # Get original dimensions
            if i < len(image_sizes):
                orig_w, orig_h = image_sizes[i]
                # Compute new width to keep aspect ratio
                new_height = min(subplot_height_px, 400)  # Cap the height
                new_width = int(orig_w * new_height / orig_h)
                
                # Convert to PIL and resize
                if img.ndim == 2:  # Grayscale
                    img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                else:  # RGB
                    img_pil = Image.fromarray((img * 255).astype(np.uint8))
                
                img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                axs[i][1].imshow(img_resized, cmap=cmap)
            else:
                axs[i][1].imshow(img, cmap=cmap)
                
        except Exception as e:
            print(f"Warning: Could not process image at index {i}: {str(e)}")
            axs[i][1].text(0.5, 0.5, "Image Error", ha='center', va='center')
        
        axs[i][1].axis('off')
        axs[i][1].set_title("Input Image", fontsize=12)
    
    # Use try-except for tight_layout
    try:
        plt.tight_layout(pad=1.0)
    except Exception as e:
        print(f"Warning: tight_layout failed: {str(e)}")
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    if output_path is not None:
        try:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.2)
            print(f"Saved comparison image to {output_path}")
        except Exception as e:
            print(f"Warning: Could not save figure: {str(e)}")
            # Try saving without bbox_inches
            plt.savefig(output_path, dpi=dpi, pad_inches=0.2)
    
    if log_wandb:
        try:
            wandb.log({"comparison": wandb.Image(output_path if output_path else fig)})
        except Exception as e:
            print(f"Warning: Could not log to wandb: {str(e)}")
    
    plt.close(fig)





if __name__ == "__main__":
    data_cache_dir = "./im2latex_new"
    dataset = "AlFrauch/im2latex"
    if os.path.exists(data_cache_dir):
        print(f"### Loading dataset from {data_cache_dir} ###")
        subset = datasets.Dataset.load_from_disk(data_cache_dir)
    else:
        streamed_dataset = load_dataset(dataset, cache_dir="./im2latex_new", split="train", streaming=True)
        # Take first 100,000 examples and convert to Dataset
        subset = datasets.Dataset.from_list(list(islice(streamed_dataset, 200)))
        subset.save_to_disk("./im2latex_new")


    train_dataset = Im2LatexDataset(subset.select(range(200)))


    text, answer, image = train_dataset[0]
    image.show()
    to_tensor = transforms.ToTensor()
    image_tensor = to_tensor(image)
    print(f"Image tensor shape: {image_tensor.shape}")

    if image.mode != "RGB":
        image = image.convert("RGB")
        image.show()
        image_tensor_rgb = to_tensor(image)
        print(f"Converted image tensor shape: {image_tensor_rgb.shape}") 



