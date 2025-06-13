from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler
from torch.optim import AdamW
from utils import Im2LatexDataset
from dataclasses import dataclass
from typing import Optional
import tyro
from lora import LoraConfig, LoraModel
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
import pdb
import wandb
from itertools import islice
import datasets


class Trainer:
    def __init__(self,train_loader, val_loader, model, processor, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.processor = processor
        self.processor = self._setup_tokenizer_properly(processor)
        self.config = config
        
        # Check if model is wrapped with DataParallel
        self.is_parallel = isinstance(model, torch.nn.DataParallel)

        # define accuracy metrics
        vocab_size = self.processor.tokenizer.vocab_size
        self.token_accuracy_metric = Accuracy(task="multiclass", num_classes=vocab_size).to(self.config.device)
        self.sequence_accuracy_metric = Accuracy(task="binary").to(self.config.device)

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.num_training_steps = config.epochs * len(train_loader)
        self.lr_scheduler = get_scheduler(
            name=self.config.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
    
    def _setup_tokenizer_properly(self, processor):
        """Ensure tokenizer is set up correctly for training"""
        
        # Make sure we have proper special tokens
        if processor.tokenizer.pad_token is None:
            # Use EOS as pad token if no pad token exists
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        
        # Ensure the model knows about padding
        if hasattr(processor.tokenizer, 'padding_side'):
            processor.tokenizer.padding_side = 'right'  # Pad on the right
        
        return processor


    def train(self):
        print("#####  Starting training...  #####")
        train_global_step = 0
        self.val_global_step = 0
        for epoch in range(self.config.epochs):
            self.model.train()

            train_loss = 0
            train_token_accuracy = 0.0
            train_sequence_accuracy = 0.0

            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}/{self.config.epochs}")):
                inputs, answers, image_sizes = batch # with collate_fn

                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = self.processor.tokenizer(
                        text=answers, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=1024,
                        add_special_tokens=True,  # Ensure EOS token is added
                        return_token_type_ids=False
                    ).input_ids.to(self.config.device)
    
                outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # Handle DataParallel loss aggregation
                if self.is_parallel:
                    loss = loss.mean()  # Average the loss across GPUs
                
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # compute token level accuracy
                token_accuracy = self._calculate_token_accuracy(outputs, labels)
                sequence_accuracy = self._calculate_sequence_accuracy(outputs, labels)

                train_token_accuracy += token_accuracy.item()
                train_sequence_accuracy += sequence_accuracy.item()

                if (batch_idx + 1) % 10 == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "train_token_accuracy": token_accuracy.item(),
                        "train_sequence_accuracy": sequence_accuracy.item(),
                        "train_global_step": train_global_step
                    }, step=train_global_step)

                    length = min(5, len(answers))
                    codes = self._eval_run_example(inputs, image_sizes, length=length)
                    # for i, (code, ans) in enumerate(zip(test_res, answers)):
                    if (batch_idx + 1) % 100 == 0:
                        for i in range(length):
                            print(f"#### latex_code_{i}: {codes[i]} ####")
                            print(f"#### ground_truth_{i}: {answers[i]} ####")

                if (batch_idx + 1) % 500 == 0:
                    self._validate(epoch)  

                train_global_step += 1  # increment after each batch

            avg_train_loss = train_loss / (batch_idx + 1)  # Fixed: use batch_idx + 1
            avg_train_token_accuracy = train_token_accuracy / (batch_idx + 1)
            avg_train_sequence_accuracy = train_sequence_accuracy / (batch_idx + 1)
            print(f"Epoch {epoch + 1}/{self.config.epochs} - "
                  f"Loss: {avg_train_loss:.4f}, "
                  f"Token Accuracy: {avg_train_token_accuracy:.4f}, "
                  f"Sequence Accuracy: {avg_train_sequence_accuracy:.4f}")

            # Log training metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_epoch_loss": avg_train_loss,
                "train_epoch_token_accuracy": avg_train_token_accuracy,
                "train_epoch_sequence_accuracy": avg_train_sequence_accuracy
            }, step=train_global_step)

            # # Validation phase
            # self._validate(epoch)
            
            output_dir = f"./checkpoints/epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)

            # Save model + processor locally
            model_path = os.path.join(output_dir, "model.safetensors")
            # Access the actual model if wrapped in DataParallel
            model_to_save = self.model.module if self.is_parallel else self.model
            model_to_save.save_model(model_path)
            self.processor.save_pretrained(output_dir)

            # Log as wandb artifact
            artifact = wandb.Artifact(
                name=f"model-epoch-{epoch+1}",
                type="model",
                metadata={"epoch": epoch+1}
            )
            artifact.add_file(model_path)
            artifact.add_dir(output_dir)  # includes tokenizer config, vocab, etc.

            wandb.log_artifact(artifact)

    def _validate(self,epoch):
        self.model.eval()
        val_loss = 0
        val_token_accuracy = 0.0
        val_sequence_accuracy = 0.0
        self.token_accuracy_metric.reset()
        self.sequence_accuracy_metric.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Validation Epoch {epoch + 1}/{self.config.epochs}")):
                inputs, answers, image_sizes = batch # with collate_fn
            
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                labels = self.processor.tokenizer(
                        text=answers, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=1024,
                        add_special_tokens=True,  # Ensure EOS token is added
                        return_token_type_ids=False
                    ).input_ids.to(self.config.device)
    
                outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                
                # Handle DataParallel loss aggregation
                if self.is_parallel:
                    loss = loss.mean()  # Average the loss across GPUs
                
                val_loss += loss.item()

                self.val_global_step += 1  # increment after each batch
                
                # compute token level accuracy 
                token_accuracy = self._calculate_token_accuracy(outputs, labels)
                sequence_accuracy = self._calculate_sequence_accuracy(outputs, labels)
                val_token_accuracy += token_accuracy.item()
                val_sequence_accuracy += sequence_accuracy.item()

                if batch_idx % 5 == 0:
                    wandb.log({
                        "val_loss": loss.item(),
                        "val_token_accuracy": token_accuracy.item(),
                        "val_sequence_accuracy": sequence_accuracy.item(),
                        "val_global_step": self.val_global_step
                    })
            
        length = min(5, len(answers))
        codes = self._eval_run_example(inputs, image_sizes, length=length)
        for i in range(length):
            print(f"#### latex_code_{i}: {codes[i]} ####")
            print(f"#### ground_truth_{i}: {answers[i]} ####")
        
        # render(test_res, pixel_values, image_sizes,output_path="comparison.jpg", log_wandb=True)

        avg_val_loss = val_loss / (batch_idx + 1)  # Fixed: use batch_idx + 1
        avg_val_token_accuracy = val_token_accuracy / (batch_idx + 1)
        avg_val_sequence_accuracy = val_sequence_accuracy / (batch_idx + 1)
        print(f"Validation - "
              f"Loss: {avg_val_loss:.4f}, "
              f"Token Accuracy: {avg_val_token_accuracy:.4f}, "
              f"Sequence Accuracy: {avg_val_sequence_accuracy:.4f}")
        # Log validation metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "val_epoch_loss": avg_val_loss,
            "val_epoch_token_accuracy": avg_val_token_accuracy,
            "val_epoch_sequence_accuracy": avg_val_sequence_accuracy
        }, step=self.val_global_step)

    def _eval_run_example(self, inputs, image_sizes, length=1):
        task_prompt = "<OCR>"
        
        # Access the actual model if wrapped in DataParallel
        if self.is_parallel:
            generated_ids = self.model.module.lora_model.generate(
                input_ids=inputs["input_ids"][:length],  # Only process the requested length
                pixel_values=inputs["pixel_values"][:length],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                early_stopping=True,  # Stop at EOS token
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                use_cache=True,
                # Remove invalid values to prevent generation issues
                remove_invalid_values=True
            )
        else:
            generated_ids = self.model.lora_model.generate(
                input_ids=inputs["input_ids"][:length],  # Only process the requested length
                pixel_values=inputs["pixel_values"][:length],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
                early_stopping=True,  # Stop at EOS token
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                use_cache=True,
                # Remove invalid values to prevent generation issues
                remove_invalid_values=True
            )


        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        answers = []
        for i in range(length):
            parsed_answer = self.processor.post_process_generation(generated_text[i], task=task_prompt, image_size=image_sizes[i])
            answers.append(parsed_answer)

        return answers
    
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

@dataclass
class Config:
    cuda: bool = True
    wandb_project_name: str = "VLM-IM2LATEX"
    checkpoint_path: Optional[str] = None

    learning_rate: float = 1e-6
    epochs: int = 10
    batch_size: int = 16
    num_workers: int = 0
    num_warmup_steps: int = 2000
    scheduler_type: str = "linear"
    device: str = "cuda"
    dataset: str = "AlFrauch/im2latex"

    use_lora = True
    train_head_only = False
    target_modules = ['q_proj', 'k_proj', 'v_proj','lm_head', 'fc2', 'fc1']
    rank = 8
    lora_alpha = 16
    use_rslora = True
    bias = "none"
    lora_dropout = 0.1
    vlm_model: str = "microsoft/Florence-2-base-ft"
    date_cache_dir: str = "./im2latex_new"
    image_size: int = 680

if __name__ == "__main__":
    config = tyro.cli(Config)
    if config.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"### Using device: {device} ###")
    print(f"### Available GPUs: {torch.cuda.device_count()} ###")
    config.device = device
    print(f"### using {config.dataset} dataset ###")
    if os.path.exists(config.date_cache_dir):
        print(f"### Loading dataset from {config.date_cache_dir} ###")
        subset = datasets.Dataset.load_from_disk(config.date_cache_dir)
    else:
        streamed_dataset = load_dataset(config.dataset, cache_dir="./im2latex_new", split="train", streaming=True)
        # Take first 100,000 examples and convert to Dataset
        subset = datasets.Dataset.from_list(list(islice(streamed_dataset, 130000)))
        subset.save_to_disk("./im2latex_new")
    
    # Split 10% from train → test
    split1 = subset.train_test_split(test_size=0.10, seed=42)
    train_temp = split1["train"]
    test_set = split1["test"]

    split2 = train_temp.train_test_split(test_size=0.10 / 0.90, seed=42)
    train_set = split2["train"]
    val_set = split2["test"]

    data = {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }

    print(f"### loading model {config.vlm_model} ###")
    model = AutoModelForCausalLM.from_pretrained(config.vlm_model, trust_remote_code=True, revision='refs/pr/6').to(device)
    processor = AutoProcessor.from_pretrained(config.vlm_model, trust_remote_code=True, revision='refs/pr/6')
    torch.cuda.empty_cache()

    ### LoRA Setups ###
    if config.use_lora:
        print("### Converting pretrained model into LoRA ###")
        lora_config = LoraConfig(
            rank=config.rank, 
            target_modules=config.target_modules, 
            exclude_modules=config.exclude_modules, 
            lora_alpha=config.lora_alpha, 
            lora_dropout=config.lora_dropout, 
            bias=config.bias, 
            use_rslora=config.use_rslora
        )
        model = LoraModel(model, lora_config).to(device)

    # Enable multi-GPU training if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"### Using {torch.cuda.device_count()} GPUs for training ###")
        model = torch.nn.DataParallel(model)

    def collate_fn(batch):
        questions, answers, images = zip(*batch)
        img_sizes = [img.size for img in images]  # Keep original sizes

        # Resize and normalize images with image processor
        pixel_values = processor.image_processor(
                                list(images),
                                size={"height": config.image_size, "width": config.image_size},
                                return_tensors="pt"
                            )  # [B, 3, H, W]

        # Tokenize text ONLY
        encoded_text = processor.tokenizer(
            list(questions),
            padding=True,
            return_tensors="pt",
            truncation=True
        )
        # Combine manually
        inputs = {"input_ids": encoded_text["input_ids"].to(device),
                  "attention_mask": encoded_text["attention_mask"].to(device),
                  "pixel_values": pixel_values["pixel_values"].to(device)}
        return inputs, answers, img_sizes



    train_dataset = Im2LatexDataset(data['train'].select(range(50_000)))
    val_dataset = Im2LatexDataset(data['val'].select(range(2000)))
    
    # Force num_workers=0 to avoid CUDA multiprocessing issues
    effective_num_workers = 0 if torch.cuda.is_available() else config.num_workers
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=effective_num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=effective_num_workers)

    wandb.init(project=config.wandb_project_name, config=config)
    trainer = Trainer(train_loader, val_loader, model, processor, config=config)
    trainer.train()
