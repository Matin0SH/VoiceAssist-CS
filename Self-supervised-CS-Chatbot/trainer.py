# """
# PyTorch-Only Customer Service Chatbot Trainer
# This script trains a customer service chatbot using PyTorch only (no TensorFlow dependencies)
# """

# import os
# import json
# import torch
# import numpy as np
# import pandas as pd
# import logging
# import time
# import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# from peft import (
#     prepare_model_for_kbit_training,
#     LoraConfig,
#     get_peft_model,
#     PeftModel,
#     PeftConfig
# )
# import argparse
# from tqdm import tqdm
# import random
# import csv
# import nltk
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize
# from rouge_score import rouge_scorer
# import re
# import math

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("chatbot_training.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)


# # Set seed for reproducibility
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     logger.info(f"Set random seed to {seed}")


# class CustomerServiceDataset(Dataset):
#     """
#     Dataset for customer service chatbot training
#     """
#     def __init__(self, data_path, tokenizer, max_length=512, format_type="instruction"):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.format_type = format_type
        
#         # Load data
#         logger.info(f"Loading dataset from {data_path}")
#         with open(data_path, 'r', encoding='utf-8') as f:
#             self.data = json.load(f)
        
#         logger.info(f"Loaded {len(self.data)} examples")
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         item = self.data[idx]
        
#         if self.format_type in ["instruction", "context"]:
#             # Format for instruction fine-tuning
#             instruction = item["instruction"]
#             input_text = item["input"]
#             output = item["output"]
            
#             if input_text:
#                 prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output}</s>"
#             else:
#                 prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
        
#         elif self.format_type == "chat":
#             # Format for chat fine-tuning
#             messages = item["messages"]
#             prompt = "<s>"
            
#             for message in messages:
#                 role = message["role"]
#                 content = message["content"]
                
#                 if role == "user":
#                     prompt += f"[INST] {content} [/INST] "
#                 elif role == "assistant":
#                     prompt += f"{content}</s>"
        
#         # Tokenize
#         encodings = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")
#         encodings["labels"] = encodings["input_ids"].copy()
        
#         # Create attention mask
#         encodings["attention_mask"] = encodings["attention_mask"]
        
#         return {
#             "input_ids": torch.tensor(encodings["input_ids"]),
#             "attention_mask": torch.tensor(encodings["attention_mask"]),
#             "labels": torch.tensor(encodings["labels"])
#         }


# def load_tokenizer_and_model(model_name, use_4bit=True, use_8bit=False):
#     """
#     Load tokenizer and model
#     """
#     logger.info(f"Loading model: {model_name}")
#     start_time = time.time()
    
#     # Configure quantization
#     compute_dtype = torch.bfloat16
    
#     if use_4bit:
#         logger.info("Using 4-bit quantization")
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=compute_dtype,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True
#         )
#     elif use_8bit:
#         logger.info("Using 8-bit quantization")
#         quantization_config = BitsAndBytesConfig(
#             load_in_8bit=True
#         )
#     else:
#         logger.info("Using full precision")
#         quantization_config = None
    
#     # Load tokenizer
#     logger.info("Loading tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     tokenizer.pad_token = tokenizer.eos_token
    
#     # Load model
#     logger.info("Loading model...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=quantization_config,
#         torch_dtype=compute_dtype,
#         device_map="auto"
#     )
    
#     # Prepare model for k-bit training if quantized
#     if use_4bit or use_8bit:
#         logger.info("Preparing model for k-bit training")
#         model = prepare_model_for_kbit_training(model)
    
#     elapsed_time = time.time() - start_time
#     logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
    
#     return tokenizer, model


# def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=None):
#     """
#     Setup LoRA for parameter-efficient fine-tuning
#     """
#     logger.info("Setting up LoRA for fine-tuning")
    
#     # Default target modules for common models
#     if target_modules is None:
#         # These work for Llama and similar models
#         target_modules = [
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj"
#         ]
    
#     # Create LoRA config
#     lora_config = LoraConfig(
#         r=r,
#         lora_alpha=lora_alpha,
#         target_modules=target_modules,
#         lora_dropout=lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM"
#     )
    
#     logger.info(f"LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
#     logger.info(f"Target modules: {target_modules}")
    
#     # Get PEFT model
#     model = get_peft_model(model, lora_config)
    
#     # Log trainable parameters
#     model.print_trainable_parameters()
    
#     return model


# class CustomTrainer(Trainer):
#     """
#     Custom trainer to log metrics and evaluate models
#     """
#     def __init__(self, *args, log_dir=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.log_dir = log_dir or "./logs"
#         os.makedirs(self.log_dir, exist_ok=True)
        
#         # Initialize metrics tracking
#         self.train_metrics = []
#         self.eval_metrics = []
        
#         # Create CSV files for metrics
#         self.train_metrics_file = os.path.join(self.log_dir, "train_metrics.csv")
#         self.eval_metrics_file = os.path.join(self.log_dir, "eval_metrics.csv")
        
#         with open(self.train_metrics_file, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['step', 'loss', 'learning_rate', 'epoch'])
        
#         with open(self.eval_metrics_file, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(['step', 'loss', 'perplexity'])

#     def log(self, logs):
#         """Override log method to save metrics"""
#         logs = super().log(logs)
        
#         # Extract metrics
#         step = self.state.global_step
#         loss = logs.get('loss', None)
#         lr = logs.get('learning_rate', None)
#         epoch = self.state.epoch
        
#         # Save training metrics
#         if loss is not None:
#             self.train_metrics.append({
#                 'step': step,
#                 'loss': loss,
#                 'learning_rate': lr,
#                 'epoch': epoch
#             })
            
#             with open(self.train_metrics_file, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([step, loss, lr, epoch])
        
#         return logs
    
#     def evaluate(self, *args, **kwargs):
#         """Override evaluate method to save metrics"""
#         output = super().evaluate(*args, **kwargs)
        
#         # Extract metrics
#         step = self.state.global_step
#         loss = output.get('eval_loss', None)
#         perplexity = math.exp(loss) if loss is not None else None
        
#         # Save evaluation metrics
#         if loss is not None:
#             self.eval_metrics.append({
#                 'step': step,
#                 'loss': loss,
#                 'perplexity': perplexity
#             })
            
#             with open(self.eval_metrics_file, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow([step, loss, perplexity])
            
#             logger.info(f"Step {step}: eval_loss = {loss:.4f}, perplexity = {perplexity:.4f}")
        
#         return output
    
#     def plot_metrics(self):
#         """Create plots for metrics"""
#         logger.info("Generating training plots...")
        
#         # Convert to DataFrames
#         train_df = pd.DataFrame(self.train_metrics)
#         eval_df = pd.DataFrame(self.eval_metrics)
        
#         if not train_df.empty:
#             # Plot training loss
#             plt.figure(figsize=(10, 6))
#             plt.plot(train_df['step'], train_df['loss'])
#             plt.title('Training Loss vs Steps')
#             plt.xlabel('Steps')
#             plt.ylabel('Loss')
#             plt.grid(True)
#             plt.savefig(os.path.join(self.log_dir, 'training_loss.png'))
#             plt.close()
            
#             # Plot learning rate
#             plt.figure(figsize=(10, 6))
#             plt.plot(train_df['step'], train_df['learning_rate'])
#             plt.title('Learning Rate vs Steps')
#             plt.xlabel('Steps')
#             plt.ylabel('Learning Rate')
#             plt.grid(True)
#             plt.savefig(os.path.join(self.log_dir, 'learning_rate.png'))
#             plt.close()
        
#         if not eval_df.empty:
#             # Plot evaluation loss
#             plt.figure(figsize=(10, 6))
#             plt.plot(eval_df['step'], eval_df['loss'])
#             plt.title('Evaluation Loss vs Steps')
#             plt.xlabel('Steps')
#             plt.ylabel('Loss')
#             plt.grid(True)
#             plt.savefig(os.path.join(self.log_dir, 'eval_loss.png'))
#             plt.close()
            
#             # Plot perplexity
#             plt.figure(figsize=(10, 6))
#             plt.plot(eval_df['step'], eval_df['perplexity'])
#             plt.title('Perplexity vs Steps')
#             plt.xlabel('Steps')
#             plt.ylabel('Perplexity')
#             plt.grid(True)
#             plt.savefig(os.path.join(self.log_dir, 'perplexity.png'))
#             plt.close()


# def compute_metrics(pred_texts, ref_texts):
#     """
#     Compute evaluation metrics for generated responses
#     """
#     metrics = {}
    
#     # 1. BLEU scores
#     bleu_scores = {f'bleu-{i+1}': [] for i in range(4)}
#     smoothie = SmoothingFunction().method1
    
#     for pred, ref in zip(pred_texts, ref_texts):
#         pred_tokens = word_tokenize(pred.lower())
#         ref_tokens = [word_tokenize(ref.lower())]
        
#         for n in range(4):
#             if len(pred_tokens) > 0 and len(ref_tokens[0]) > 0:
#                 try:
#                     score = sentence_bleu(ref_tokens, pred_tokens, 
#                                          weights=([1.0/float(n+1)] * (n+1) + [0] * (4-(n+1))),
#                                          smoothing_function=smoothie)
#                     bleu_scores[f'bleu-{n+1}'].append(score)
#                 except Exception as e:
#                     logger.warning(f"Error computing BLEU-{n+1}: {e}")
#                     bleu_scores[f'bleu-{n+1}'].append(0.0)
#             else:
#                 bleu_scores[f'bleu-{n+1}'].append(0.0)
    
#     # Average BLEU scores
#     for k, scores in bleu_scores.items():
#         if scores:
#             metrics[k] = sum(scores) / len(scores)
    
#     # 2. ROUGE scores
#     rouge_types = ['rouge1', 'rouge2', 'rougeL']
#     rouge_results = {rouge_type: [] for rouge_type in rouge_types}
#     scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
#     for pred, ref in zip(pred_texts, ref_texts):
#         try:
#             scores = scorer.score(ref, pred)
#             for rouge_type in rouge_types:
#                 rouge_results[rouge_type].append(scores[rouge_type].fmeasure)
#         except Exception as e:
#             logger.warning(f"Error computing ROUGE: {e}")
#             for rouge_type in rouge_types:
#                 rouge_results[rouge_type].append(0.0)
    
#     # Average ROUGE scores
#     for rouge_type, scores in rouge_results.items():
#         if scores:
#             metrics[rouge_type] = sum(scores) / len(scores)
    
#     # 3. Response statistics
#     metrics['avg_response_length'] = sum(len(word_tokenize(pred)) for pred in pred_texts) / len(pred_texts) if pred_texts else 0
#     metrics['lexical_diversity'] = sum(len(set(word_tokenize(pred.lower()))) / max(1, len(word_tokenize(pred))) for pred in pred_texts) / len(pred_texts) if pred_texts else 0
    
#     return metrics


# def train_model(model, tokenizer, train_data_path, val_data_path, output_dir, 
#                 format_type="instruction", batch_size=4, num_epochs=3, 
#                 learning_rate=2e-4, max_length=512, log_dir=None):
#     """
#     Train the model using the provided data with comprehensive logging
#     """
#     # Setup logging directory
#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     if log_dir is None:
#         log_dir = os.path.join(output_dir, f"logs_{timestamp}")
    
#     os.makedirs(log_dir, exist_ok=True)
#     logger.info(f"Logs will be saved to {log_dir}")
    
#     # Create datasets
#     logger.info(f"Creating datasets from {train_data_path} and {val_data_path}")
#     train_dataset = CustomerServiceDataset(
#         train_data_path, tokenizer, max_length=max_length, format_type=format_type
#     )
    
#     val_dataset = CustomerServiceDataset(
#         val_data_path, tokenizer, max_length=max_length, format_type=format_type
#     )
    
#     logger.info(f"Train dataset size: {len(train_dataset)}")
#     logger.info(f"Validation dataset size: {len(val_dataset)}")
    
#     # Calculate training steps
#     gradient_accumulation_steps = 4
#     num_train_epochs = num_epochs
#     train_steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
#     num_training_steps = train_steps_per_epoch * num_train_epochs
    
#     logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
#     logger.info(f"Total training steps: {num_training_steps}")
    
#     # Create training arguments
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=num_epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         gradient_checkpointing=True,
#         evaluation_strategy="steps",
#         eval_steps=train_steps_per_epoch // 2,  # Evaluate twice per epoch
#         save_strategy="steps",
#         save_steps=train_steps_per_epoch // 2,  # Save twice per epoch
#         save_total_limit=3,
#         learning_rate=learning_rate,
#         weight_decay=0.01,
#         adam_beta2=0.95,
#         warmup_ratio=0.1,
#         lr_scheduler_type="cosine",
#         logging_dir=os.path.join(log_dir, "hf_logs"),
#         logging_steps=50,
#         report_to="none",  # Disable TensorBoard which requires TensorFlow
#         fp16=torch.cuda.is_available(),
#         bf16=False,  # Set to True if your GPU supports it
#         push_to_hub=False,
#         remove_unused_columns=False,
#         group_by_length=True,
#         dataloader_num_workers=4,
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         greater_is_better=False
#     )
    
#     # Create data collator
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, 
#         mlm=False
#     )
    
#     # Create trainer
#     trainer = CustomTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#         log_dir=log_dir
#     )
    
#     # Show model size
#     model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
#     logger.info(f"Trainable model size: {model_size:.2f}M parameters")
    
#     # Show example batch
#     sample_batch = next(iter(DataLoader(train_dataset, batch_size=1)))
#     logger.info("\nSample input:")
#     decoded_text = tokenizer.decode(sample_batch["input_ids"][0])
#     logger.info(decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text)
    
#     # Train model
#     logger.info("Starting training...")
#     start_time = time.time()
    
#     try:
#         train_result = trainer.train()
        
#         # Log training metrics
#         train_metrics = train_result.metrics
#         trainer.log_metrics("train", train_metrics)
#         trainer.save_metrics("train", train_metrics)
#     except Exception as e:
#         logger.error(f"Training error: {e}")
#         raise
    
#     # Training time
#     training_time = time.time() - start_time
#     logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
#     # Log evaluation metrics
#     logger.info("Final evaluation...")
#     eval_metrics = trainer.evaluate()
#     trainer.log_metrics("eval", eval_metrics)
#     trainer.save_metrics("eval", eval_metrics)
    
#     # Plot metrics
#     trainer.plot_metrics()
    
#     # Save final model
#     logger.info("Saving model...")
#     trainer.save_model(output_dir)
    
#     # Save tokenizer for ease of use later
#     tokenizer.save_pretrained(output_dir)
    
#     return model, tokenizer, train_metrics, eval_metrics


# def generate_responses(model, tokenizer, test_data_path, format_type="instruction", 
#                       max_length=512, num_samples=None, temperature=0.7, top_p=0.9,
#                       output_dir=None):
#     """
#     Generate responses for test data and evaluate
#     """
#     logger.info(f"Generating responses on {test_data_path}")
    
#     # Create output directory if not provided
#     if output_dir is None:
#         timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#         output_dir = f"./evaluation_{timestamp}"
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load test data
#     with open(test_data_path, 'r', encoding='utf-8') as f:
#         test_data = json.load(f)
    
#     # Limit samples if specified
#     if num_samples is not None and num_samples < len(test_data):
#         logger.info(f"Using {num_samples} samples out of {len(test_data)}")
#         test_data = random.sample(test_data, num_samples)
#     else:
#         logger.info(f"Evaluating on all {len(test_data)} samples")
    
#     all_prompts = []
#     all_generated = []
#     all_references = []
#     generation_times = []
    
#     # Generate responses for each test example
#     for idx, item in enumerate(tqdm(test_data, desc="Generating responses")):
#         if format_type in ["instruction", "context"]:
#             instruction = item["instruction"]
#             input_text = item["input"]
#             reference_output = item["output"]
            
#             if input_text:
#                 prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST] "
#             else:
#                 prompt = f"<s>[INST] {instruction} [/INST] "
        
#         elif format_type == "chat":
#             messages = item["messages"]
#             prompt = "<s>"
#             reference_output = ""
            
#             for i, message in enumerate(messages):
#                 role = message["role"]
#                 content = message["content"]
                
#                 if role == "user":
#                     prompt += f"[INST] {content} [/INST] "
#                 elif role == "assistant":
#                     if i == len(messages) - 1:  # Last message is the one we want to generate
#                         reference_output = content
#                     else:
#                         prompt += f"{content}</s>"
        
#         # Tokenize
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
#         # Generate response
#         start_time = time.time()
#         try:
#             with torch.no_grad():
#                 outputs = model.generate(
#                     input_ids=inputs["input_ids"],
#                     attention_mask=inputs["attention_mask"],
#                     max_new_tokens=256,
#                     temperature=temperature,
#                     top_p=top_p,
#                     do_sample=True,
#                     pad_token_id=tokenizer.eos_token_id
#                 )
            
#             # Measure generation time
#             gen_time = time.time() - start_time
#             generation_times.append(gen_time)
            
#             # Decode response
#             response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
#             # Extract the generated response (remove the input prompt)
#             generated_response = response[len(prompt):]
            
#             # Clean up any remaining special tokens or formatting
#             generated_response = generated_response.replace("</s>", "").strip()
            
#             all_prompts.append(prompt)
#             all_generated.append(generated_response)
#             all_references.append(reference_output)
            
#             # Log samples periodically
#             if idx % 10 == 0 or idx < 5:
#                 logger.info(f"\nSample {idx+1}:")
#                 logger.info(f"Prompt: {prompt[:100]}...")
#                 logger.info(f"Generated: {generated_response[:100]}...")
#                 logger.info(f"Reference: {reference_output[:100]}...")
#                 logger.info(f"Generation time: {gen_time:.2f}s")
        
#         except Exception as e:
#             logger.error(f"Error generating response for sample {idx}: {e}")
    
#     # Compute metrics
#     logger.info("Computing evaluation metrics...")
#     metrics = compute_metrics(all_generated, all_references)
    
#     # Add generation time metrics
#     metrics['avg_generation_time'] = np.mean(generation_times)
#     metrics['min_generation_time'] = np.min(generation_times)
#     metrics['max_generation_time'] = np.max(generation_times)
    
#     # Log metrics
#     logger.info("\nEvaluation Metrics:")
#     for metric_name, metric_value in metrics.items():
#         logger.info(f"{metric_name}: {metric_value:.4f}")
    
#     # Save metrics to file
#     metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
#     with open(metrics_file, 'w', encoding='utf-8') as f:
#         json.dump(metrics, f, indent=2)
#     logger.info(f"Metrics saved to {metrics_file}")
    
#     # Save generated responses
#     results_file = os.path.join(output_dir, "evaluation_results.jsonl")
#     with open(results_file, 'w', encoding='utf-8') as f:
#         for prompt, generated, reference in zip(all_prompts, all_generated, all_references):
#             result = {
#                 "prompt": prompt,
#                 "generated": generated,
#                 "reference": reference
#             }
#             f.write(json.dumps(result) + "\n")
#     logger.info(f"Results saved to {results_file}")
    
#     # Create visualizations
#     logger.info("Creating visualizations...")
    
#     # 1. Metrics bar chart
#     plt.figure(figsize=(14, 6))
#     metrics_to_plot = {k: v for k, v in metrics.items() if k not in ['avg_generation_time', 'min_generation_time', 'max_generation_time']}
#     plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
#     plt.title('Evaluation Metrics')
#     plt.ylabel('Score')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, "metrics_bar_chart.png"))
#     plt.close()
    
#     # 2. Response length distribution
#     plt.figure(figsize=(10, 6))
#     generated_lengths = [len(word_tokenize(text)) for text in all_generated]
#     reference_lengths = [len(word_tokenize(text)) for text in all_references]
    
#     plt.hist(generated_lengths, alpha=0.5, label='Generated', bins=30)
#     plt.hist(reference_lengths, alpha=0.5, label='Reference', bins=30)
#     plt.title('Response Length Distribution')
#     plt.xlabel('Length (words)')
#     plt.ylabel('Count')
#     plt.legend()
#     plt.savefig(os.path.join(output_dir, "response_length_distribution.png"))
#     plt.close()
    
#     # 3. Generation time distribution
#     plt.figure(figsize=(10, 6))
#     plt.hist(generation_times, bins=20)
#     plt.title('Generation Time Distribution')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Count')
#     plt.savefig(os.path.join(output_dir, "generation_time_distribution.png"))
#     plt.close()
    
#     # Create a comprehensive report
#     report_file = os.path.join(output_dir, "evaluation_report.md")
#     with open(report_file, 'w', encoding='utf-8') as f:
#         f.write("# Model Evaluation Report\n\n")
        
#         f.write("## Overview\n\n")
#         f.write(f"- Model: {model.config._name_or_path}\n")
#         f.write(f"- Format type: {format_type}\n")
#         f.write(f"- Number of test examples: {len(all_generated)}\n")
#         f.write(f"- Evaluation date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
#         f.write("## Metrics Summary\n\n")
#         f.write("| Metric | Value |\n")
#         f.write("|--------|-------|\n")
#         for metric_name, metric_value in metrics.items():
#             f.write(f"| {metric_name} | {metric_value:.4f} |\n")
#         f.write("\n")
        
#         f.write("## Response Examples\n\n")
#         for i in range(min(5, len(all_generated))):
#             f.write(f"### Example {i+1}\n\n")
#             f.write("**Input:**\n\n")
#             f.write(f"```\n{all_prompts[i]}\n```\n\n")
#             f.write("**Generated:**\n\n")
#             f.write(f"```\n{all_generated[i]}\n```\n\n")
#             f.write("**Reference:**\n\n")
#             f.write(f"```\n{all_references[i]}\n```\n\n")
        
#         f.write("## Analysis\n\n")
#         f.write("### Performance Analysis\n\n")
#         f.write(f"- The model achieved a BLEU-1 score of {metrics.get('bleu-1', 'N/A'):.4f}, indicating {'good' if metrics.get('bleu-1', 0) > 0.3 else 'moderate' if metrics.get('bleu-1', 0) > 0.1 else 'poor'} lexical overlap with reference responses.\n")
#         f.write(f"- ROUGE-L score of {metrics.get('rougeL', 'N/A'):.4f} shows {'strong' if metrics.get('rougeL', 0) > 0.4 else 'moderate' if metrics.get('rougeL', 0) > 0.2 else 'limited'} overlap in the longest common subsequence.\n")
#         f.write(f"- Average response length is {metrics.get('avg_response_length', 'N/A'):.1f} words, which is {'longer' if metrics.get('avg_response_length', 0) > np.mean(reference_lengths) else 'shorter'} than the average reference length of {np.mean(reference_lengths):.1f} words.\n")
#         f.write(f"- The model generates responses in {metrics.get('avg_generation_time', 'N/A'):.2f} seconds on average.\n\n")
        
#         f.write("### Areas for Improvement\n\n")
#         # Identify areas for improvement based on metrics
#         low_metrics = []
#         if metrics.get('bleu-1', 0) < 0.2:
#             low_metrics.append("BLEU")
#         if metrics.get('rougeL', 0) < 0.3:
#             low_metrics.append("ROUGE")
        
#         if low_metrics:
#             f.write(f"- Model shows room for improvement in {', '.join(low_metrics)}.\n")
        
#         if abs(metrics.get('avg_response_length', 0) - np.mean(reference_lengths)) > 5:
#             f.write(f"- Response length distribution differs from reference responses, suggesting potential issues with verbosity or brevity.\n")
        
#         f.write("\n## Conclusion\n\n")
#         avg_score = np.mean([
#             metrics.get('bleu-1', 0),
#             metrics.get('rougeL', 0)
#         ])
        
#         if avg_score > 0.5:
#             f.write("The model performs well overall, showing strong alignment with reference responses while maintaining reasonable generation speed.\n")
#         elif avg_score > 0.3:
#             f.write("The model shows moderate performance. While it captures some aspects of the reference responses, there is room for improvement in generating more accurate and semantically similar outputs.\n")
#         else:
#             f.write("The model's performance is below expectations. Significant improvements are needed to better align with reference responses and improve overall quality.\n")
    
#     logger.info(f"Evaluation report created at {report_file}")
    
#     return metrics


# def main():
#     """
#     Main function
#     """
#     parser = argparse.ArgumentParser(description="Train a customer service chatbot with PyTorch only")
#     parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", 
#                         help="Base model to fine-tune (e.g., meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1)")
#     parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed training data")
#     parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Output directory for model")
#     parser.add_argument("--format_type", type=str, default="instruction", choices=["instruction", "chat", "context"], 
#                         help="Format type of the training data")
#     parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
#     parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
#     parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
#     parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
#     parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
#     parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
#     parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
#     parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
#     parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed")
#     parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on test data")
#     parser.add_argument("--model_path", type=str, help="Path to fine-tuned model for evaluation")
#     parser.add_argument("--log_dir", type=str, help="Directory for saving logs")
#     parser.add_argument("--eval_samples", type=int, help="Number of samples to use for evaluation (default: all)")
    
#     args = parser.parse_args()
    
#     # Set seed
#     set_seed(args.seed)
    
#     # Set file paths
#     train_path = os.path.join(args.data_dir, args.format_type, "train.json")
#     val_path = os.path.join(args.data_dir, args.format_type, "validation.json")
#     test_path = os.path.join(args.data_dir, args.format_type, "test.json")
    
#     # Check if files exist
#     for path, name in [(train_path, "Training"), (val_path, "Validation"), (test_path, "Test")]:
#         if not os.path.exists(path):
#             logger.error(f"{name} data not found at {path}")
#             return
    
#     if args.eval_only:
#         if not args.model_path:
#             logger.error("Please provide a model path for evaluation using --model_path")
#             return
        
#         logger.info(f"Loading model from {args.model_path} for evaluation...")
#         try:
#             config = PeftConfig.from_pretrained(args.model_path)
#             tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            
#             # Load base model
#             model = AutoModelForCausalLM.from_pretrained(
#                 config.base_model_name_or_path,
#                 torch_dtype=torch.bfloat16,
#                 device_map="auto"
#             )
            
#             # Load LoRA weights
#             model = PeftModel.from_pretrained(model, args.model_path)
            
#             # Run evaluation
#             generate_responses(
#                 model,
#                 tokenizer,
#                 test_path,
#                 format_type=args.format_type,
#                 max_length=args.max_length,
#                 num_samples=args.eval_samples,
#                 output_dir=os.path.join(args.output_dir, "evaluation")
#             )
#         except Exception as e:
#             logger.error(f"Error during evaluation: {e}")
#             raise
        
#     else:
#         try:
#             # Load tokenizer and model
#             logger.info(f"Loading base model: {args.model_name}")
#             tokenizer, model = load_tokenizer_and_model(
#                 args.model_name, 
#                 use_4bit=args.use_4bit,
#                 use_8bit=args.use_8bit
#             )
            
#             # Setup LoRA
#             logger.info("Setting up LoRA for parameter-efficient fine-tuning...")
#             model = setup_lora(
#                 model, 
#                 r=args.lora_r,
#                 lora_alpha=args.lora_alpha,
#                 lora_dropout=args.lora_dropout
#             )
            
#             # Train model
#             logger.info(f"Training model on {args.format_type} data...")
#             model, tokenizer, train_metrics, eval_metrics = train_model(
#                 model,
#                 tokenizer,
#                 train_path,
#                 val_path,
#                 args.output_dir,
#                 format_type=args.format_type,
#                 batch_size=args.batch_size,
#                 num_epochs=args.num_epochs,
#                 learning_rate=args.learning_rate,
#                 max_length=args.max_length,
#                 log_dir=args.log_dir
#             )
            
#             # Evaluate model on test data
#             logger.info("Evaluating model on test data...")
#             generate_responses(
#                 model,
#                 tokenizer,
#                 test_path,
#                 format_type=args.format_type,
#                 max_length=args.max_length,
#                 num_samples=args.eval_samples,
#                 output_dir=os.path.join(args.output_dir, "evaluation")
#             )
            
#         except Exception as e:
#             logger.error(f"Error during training or evaluation: {e}")
#             raise
    
#     logger.info("Process completed successfully!")


# if __name__ == "__main__":
#     main()



"""
Simplified Customer Service Chatbot Trainer without Quantization
This script works on Windows without bitsandbytes
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import logging
import time
import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)
import argparse
from tqdm import tqdm
import random
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed}")

class CustomerServiceDataset(Dataset):
    """Dataset for customer service chatbot training"""
    def __init__(self, data_path, tokenizer, max_length=512, format_type="instruction"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        
        # Load data
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.format_type in ["instruction", "context"]:
            # Format for instruction fine-tuning
            instruction = item["instruction"]
            input_text = item.get("input", "")
            output = item["output"]
            
            if input_text:
                prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST] {output}</s>"
            else:
                prompt = f"<s>[INST] {instruction} [/INST] {output}</s>"
        
        elif self.format_type == "chat":
            # Format for chat fine-tuning
            messages = item["messages"]
            prompt = "<s>"
            
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    prompt += f"[INST] {content} [/INST] "
                elif role == "assistant":
                    prompt += f"{content}</s>"
        
        # Tokenize
        encodings = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length")
        encodings["labels"] = encodings["input_ids"].copy()
        
        return {
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
            "labels": torch.tensor(encodings["labels"])
        }

def load_tokenizer_and_model(model_name):
    """Load tokenizer and model without quantization"""
    logger.info(f"Loading model: {model_name}")
    start_time = time.time()
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading model...")
    
    # Check available device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Determine dtype based on device
    if device == "cuda":
        # Check if VRAM is enough for fp16
        try:
            # Try to load with half precision
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Using float16 precision")
        except Exception as e:
            logger.warning(f"Could not load with float16: {e}")
            # Fall back to full precision
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            logger.info("Using float32 precision")
    else:
        # Use float32 for CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(device)
        logger.info("Using float32 precision on CPU")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
    
    return tokenizer, model

def setup_lora(model, r=8, lora_alpha=16, lora_dropout=0.05):
    """Setup LoRA for parameter-efficient fine-tuning with reduced parameters"""
    logger.info("Setting up LoRA for fine-tuning")
    
    # Detect model type to set appropriate target modules
    model_name_lower = model.config._name_or_path.lower()
    
    # Set target modules based on model type
    if "llama" in model_name_lower or "mistral" in model_name_lower:
        target_modules = ["q_proj", "v_proj"]
    elif "gpt" in model_name_lower:
        target_modules = ["c_attn"]
    else:
        # Default for unknown models
        target_modules = ["query", "value"]
    
    # Create LoRA config with fewer parameters for easier training
    lora_config = LoraConfig(
        r=r,  # Lower rank
        lora_alpha=lora_alpha,
        target_modules=target_modules,  # Target fewer modules
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    logger.info(f"LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameters
    model.print_trainable_parameters()
    
    return model

def train_model(model, tokenizer, train_data_path, val_data_path, output_dir, 
                format_type="instruction", batch_size=2, num_epochs=3, 
                learning_rate=2e-4, max_length=512, log_dir=None):
    """Train the model using the provided data"""
    # Setup logging directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(output_dir, f"logs_{timestamp}")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Logs will be saved to {log_dir}")
    
    # Create datasets
    logger.info(f"Creating datasets from {train_data_path} and {val_data_path}")
    train_dataset = CustomerServiceDataset(
        train_data_path, tokenizer, max_length=max_length, format_type=format_type
    )
    
    val_dataset = CustomerServiceDataset(
        val_data_path, tokenizer, max_length=max_length, format_type=format_type
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Reduce batch size if on CPU
    if not torch.cuda.is_available():
        batch_size = 1
        logger.warning("Running on CPU, reduced batch size to 1")
    
    # Calculate training steps
    gradient_accumulation_steps = 8  # Increased to compensate for smaller batch size
    num_train_epochs = num_epochs
    train_steps_per_epoch = max(1, len(train_dataset) // (batch_size * gradient_accumulation_steps))
    num_training_steps = train_steps_per_epoch * num_train_epochs
    
    logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
    logger.info(f"Total training steps: {num_training_steps}")
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=max(1, train_steps_per_epoch // 2),  # Evaluate twice per epoch
        save_strategy="steps",
        save_steps=max(1, train_steps_per_epoch // 2),  # Save twice per epoch
        save_total_limit=3,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_dir=os.path.join(log_dir, "hf_logs"),
        logging_steps=10,
        report_to="none",  # Disable TensorBoard
        fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA is available
        push_to_hub=False,
        remove_unused_columns=False,
        group_by_length=True,
        dataloader_num_workers=0,  # Safer for Windows
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Show model size
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    logger.info(f"Trainable model size: {model_size:.2f}M parameters")
    
    # Show example batch
    sample_batch = next(iter(DataLoader(train_dataset, batch_size=1)))
    logger.info("\nSample input:")
    decoded_text = tokenizer.decode(sample_batch["input_ids"][0])
    logger.info(decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text)
    
    # Train model
    logger.info("Starting training...")
    start_time = time.time()
    
    try:
        train_result = trainer.train()
        
        # Log training metrics
        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
    # Training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    
    # Log evaluation metrics
    logger.info("Final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    # Save final model
    logger.info("Saving model...")
    trainer.save_model(output_dir)
    
    # Save tokenizer for ease of use later
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer, train_metrics, eval_metrics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train a customer service chatbot on Windows")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Base model to fine-tune")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing processed training data")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_model", help="Output directory for model")
    parser.add_argument("--format_type", type=str, default="instruction", choices=["instruction", "chat", "context"], 
                        help="Format type of the training data")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation on test data")
    parser.add_argument("--model_path", type=str, help="Path to fine-tuned model for evaluation")
    parser.add_argument("--log_dir", type=str, help="Directory for saving logs")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set file paths
    train_path = os.path.join(args.data_dir, args.format_type, "train.json")
    val_path = os.path.join(args.data_dir, args.format_type, "validation.json")
    
    # Check if the data directory structure is correct
    if not os.path.exists(os.path.join(args.data_dir, args.format_type)):
        # Try to use multiwoz_processed.json directly
        multiwoz_processed_path = os.path.join(args.data_dir, "multiwoz_processed.json")
        if os.path.exists(multiwoz_processed_path):
            logger.info(f"Directory structure not found, but found {multiwoz_processed_path}")
            logger.info("Processing MultiWOZ data to create the required directory structure...")
            
            # Create the directory structure
            os.makedirs(os.path.join(args.data_dir, args.format_type), exist_ok=True)
            
            # Load multiwoz data
            with open(multiwoz_processed_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Convert to instruction format
            formatted_data = []
            for conversation in data:
                turns = conversation.get("turns", [])
                
                for i in range(len(turns) - 1):
                    if turns[i].get("speaker") == "user" and turns[i+1].get("speaker") == "system":
                        formatted_data.append({
                            "instruction": turns[i].get("text", ""),
                            "input": "",
                            "output": turns[i+1].get("text", "")
                        })
            
            logger.info(f"Extracted {len(formatted_data)} instruction-response pairs")
            
            # Split into train, validation, and test
            random.shuffle(formatted_data)
            train_size = int(len(formatted_data) * 0.8)
            val_size = int(len(formatted_data) * 0.1)
            
            train_data = formatted_data[:train_size]
            val_data = formatted_data[train_size:train_size+val_size]
            test_data = formatted_data[train_size+val_size:]
            
            # Save the splits
            train_path = os.path.join(args.data_dir, args.format_type, "train.json")
            val_path = os.path.join(args.data_dir, args.format_type, "validation.json")
            test_path = os.path.join(args.data_dir, args.format_type, "test.json")
            
            with open(train_path, "w", encoding="utf-8") as f:
                json.dump(train_data, f)
            
            with open(val_path, "w", encoding="utf-8") as f:
                json.dump(val_data, f)
            
            with open(test_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f)
            
            logger.info(f"Created train set with {len(train_data)} examples")
            logger.info(f"Created validation set with {len(val_data)} examples")
            logger.info(f"Created test set with {len(test_data)} examples")
        else:
            logger.error(f"Could not find {os.path.join(args.data_dir, args.format_type)} or {multiwoz_processed_path}")
            return
    
    # Check if files exist
    for path, name in [(train_path, "Training"), (val_path, "Validation")]:
        if not os.path.exists(path):
            logger.error(f"{name} data not found at {path}")
            return
    
    if args.eval_only:
        logger.error("Evaluation-only mode is not implemented in this simplified version")
        return
    
    try:
        # Load tokenizer and model
        logger.info(f"Loading base model: {args.model_name}")
        tokenizer, model = load_tokenizer_and_model(args.model_name)
        
        # Setup LoRA
        logger.info("Setting up LoRA for parameter-efficient fine-tuning...")
        model = setup_lora(
            model, 
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        
        # Train model
        logger.info(f"Training model on {args.format_type} data...")
        model, tokenizer, train_metrics, eval_metrics = train_model(
            model,
            tokenizer,
            train_path,
            val_path,
            args.output_dir,
            format_type=args.format_type,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            log_dir=args.log_dir
        )
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()