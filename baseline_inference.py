import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from dataclasses import dataclass
from typing import Tuple
from torchmetrics import Accuracy 
from itertools import islice
import datasets
import csv

class Florence2BaselineEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model, self.processor = self.load_baseline_model(config.base_model)
        
        # Initialize TorchMetrics accuracy metrics
        vocab_size = self.processor.tokenizer.vocab_size
        self.token_accuracy_metric = Accuracy(task="multiclass", num_classes=vocab_size).to(self.device)
        self.sequence_accuracy_metric = Accuracy(task="binary").to(self.device)
        
    def load_baseline_model(self, base_model_path):
        """Load baseline model without LoRA adapters"""
        print("Loading baseline model...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            trust_remote_code=True, 
            revision='refs/pr/6'
        ).to(self.device)
        
        base_model.eval()
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            base_model_path,
            trust_remote_code=True, 
            revision='refs/pr/6'
        )
        
        return base_model, processor

    def run_inference_with_logits(self, image, ground_truth_text, task_prompt="<OCR>"):
        """
        Run inference and return both generated text and logits for accuracy calculation
        
        Args:
            image: PIL Image
            ground_truth_text: Ground truth LaTeX text
            task_prompt: Task prompt (default: "<OCR>")
        
        Returns:
            Dict containing generated text, logits, and accuracy metrics
        """
        # Prepare inputs
        inputs = self.collate_fn([image], [task_prompt])

        ground_truth_label = self.processor.tokenizer(
                        text=ground_truth_text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=1024,
                        add_special_tokens=True,  # Ensure EOS token is added
                        return_token_type_ids=False
        ).input_ids.to(self.config.device)
        
        # Generate with return_dict_in_generate=True to get logits
        with torch.no_grad():
            # Forward pass to get logits results
            outputs = self.model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], labels=ground_truth_label )

            # Generate text results
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
            # Post-process
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=image.size
            )

        return outputs, ground_truth_label, parsed_answer

    def collate_fn(self, images, texts):
        image_size = 680

        # Step 1: Resize and normalize images with image processor
        pixel_values = self.processor.image_processor(
                                list(images),
                                size={"height": image_size, "width": image_size},
                                return_tensors="pt"
                            )  # [B, 3, H, W]

        # Step 2: Tokenize text ONLY
        encoded_text = self.processor.tokenizer(
            list(texts),
            padding=True,
            return_tensors="pt",
            truncation=True
        )
        
        # Step 3: Combine manually
        inputs = {"input_ids": encoded_text["input_ids"].to(self.device),
                  "attention_mask": encoded_text["attention_mask"].to(self.device),
                  "pixel_values": pixel_values["pixel_values"].to(self.device)}
        return inputs
       
    def _calculate_token_accuracy(self, outputs, labels):
        """
        Calculate token-level accuracy
        
        Args:
            outputs: model outputs with logits
            labels: target token ids [batch_size, seq_len]
        
        Returns:
            accuracy score as float
        """
        # Get predicted tokens
        pred_tokens = torch.argmax(outputs.logits, dim=-1)  # [batch_size, seq_len]
        
        # Flatten for comparison
        pred_flat = pred_tokens.view(-1)
        labels_flat = labels.view(-1)
        
        # Mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            mask = labels_flat != pad_token_id
            pred_flat = pred_flat[mask]
            labels_flat = labels_flat[mask]
        
        # Calculate accuracy
        return self.token_accuracy_metric(pred_flat, labels_flat)
    
    def _calculate_sequence_accuracy(self, outputs, labels):
        """
        Calculate sequence-level accuracy (exact match)
        
        Args:
            outputs: model outputs with logits
            labels: target token ids [batch_size, seq_len]
        
        Returns:
            sequence accuracy score as float
        """
        # Get predicted tokens
        pred_tokens = torch.argmax(outputs.logits, dim=-1)  # [batch_size, seq_len]
        
        pad_token_id = self.processor.tokenizer.pad_token_id
        batch_matches = []
        
        for pred_seq, label_seq in zip(pred_tokens, labels):
            # Remove padding for comparison
            if pad_token_id is not None:
                pred_mask = pred_seq != pad_token_id
                label_mask = label_seq != pad_token_id
                pred_clean = pred_seq[pred_mask]
                label_clean = label_seq[label_mask]
            else:
                pred_clean = pred_seq
                label_clean = label_seq
            
            # Check if sequences match exactly
            if len(pred_clean) == len(label_clean):
                exact_match = torch.all(pred_clean == label_clean)
            else:
                exact_match = torch.tensor(False)
                
            batch_matches.append(exact_match.float().to(self.config.device))
        
        # Convert to tensor and calculate accuracy
        matches = torch.stack(batch_matches)
        targets = torch.ones_like(matches) # All should be 1 for perfect match
        
        return self.sequence_accuracy_metric(matches, targets)
    
    def evaluate_dataset(self, dataset_name: str, num_samples = 100) -> Tuple[float, float]:
        """
        Evaluate the model on a HuggingFace dataset
        
        Args:
            dataset_name: Name of the HuggingFace dataset
            num_samples: Number of samples to evaluate (default: 100)
        """
        
        # dataset = load_dataset(dataset_name, split=split)
        stream_dataset = load_dataset(dataset_name, cache_dir="./im2latex_baseline", split="train", streaming=True)
        dataset = datasets.Dataset.from_list(list(islice(stream_dataset, num_samples)))
        dataset.save_to_disk("./im2latex_baseline")
        
        print(f"Dataset loaded. Evaluating on {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} samples...")
        
        
        # Reset TorchMetrics
        self.token_accuracy_metric.reset()
        self.sequence_accuracy_metric.reset()

        token_accuracy = []
        sequence_accuracy = []
        text_results = []     
        
        # Process each sample
        for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        
            image = sample["image"].convert('RGB')
            ground_truth = sample["text"]
            outputs, labels, code = self.run_inference_with_logits(image, ground_truth)

            token_level_accuracy = self._calculate_token_accuracy(outputs, labels)
            sequence_level_accuracy = self._calculate_sequence_accuracy(outputs, labels)

            token_accuracy.append(token_level_accuracy.item())
            sequence_accuracy.append(sequence_level_accuracy.item())
            text_results.append([i, ground_truth, code['<OCR>'], token_level_accuracy.item(), sequence_level_accuracy.item()])


        return token_accuracy, sequence_accuracy, text_results



@dataclass
class InfConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"    
    num_samples = 20
    base_model = "microsoft/Florence-2-base-ft"
    dataset_name = "AlFrauch/im2latex"  


if __name__ == "__main__":
    config = InfConfig()
    # Initialize evaluator
    evaluator = Florence2BaselineEvaluator(config)
    token_acc, seq_acc, text_res = evaluator.evaluate_dataset(config.dataset_name, num_samples=config.num_samples)
    print(f"Average Token Accuracy: {np.mean(token_acc):.4f}")
    print(f"Average Sequence Accuracy: {np.mean(seq_acc):.4f}")

    import csv
    output_csv_path = f"baseline_eval_{config.num_samples}_samples.csv"
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["idx", "Ground Truth", "Prediction", "Token Accuracy", "Sequence Accuracy"])
        writer.writerows(text_res)