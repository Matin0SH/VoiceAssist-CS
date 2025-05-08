"""
Customer Service Dataset Preprocessor
This script prepares datasets for training a chatbot model by:
1. Cleaning and normalizing the text
2. Converting conversations to instruction-following format
3. Generating train/validation/test splits
4. Creating output files in formats ready for model training
"""

import os
import json
import random
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm


# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def load_dataset(dataset_path):
    """
    Load a dataset from a JSON file
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_text(text):
    """
    Clean and normalize text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Replace HTML entities
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    
    # Remove URLs (simple pattern)
    text = re.sub(r'https?://\S+', '[URL]', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    return text


def extract_instruction_response_pairs(conversation):
    """
    Extract instruction-response pairs from a conversation
    Returns a list of (instruction, response) tuples
    """
    pairs = []
    turns = conversation.get('turns', [])
    
    # Skip conversations with fewer than 2 turns
    if len(turns) < 2:
        return pairs
    
    # Process the conversation to extract instruction-response pairs
    for i in range(len(turns) - 1):
        current_turn = turns[i]
        next_turn = turns[i + 1]
        
        # Check if current turn is customer and next turn is agent
        if (current_turn.get('speaker', '').lower() in ['customer', 'user'] and 
            next_turn.get('speaker', '').lower() not in ['customer', 'user']):
            
            instruction = clean_text(current_turn.get('text', ''))
            response = clean_text(next_turn.get('text', ''))
            
            # Skip empty or very short instructions/responses
            if len(instruction.split()) < 2 or len(response.split()) < 2:
                continue
            
            pairs.append((instruction, response))
    
    return pairs


def extract_multi_turn_context(conversation, max_history=3):
    """
    Extract samples with multi-turn context
    Returns a list of (context, response) tuples
    """
    samples = []
    turns = conversation.get('turns', [])
    
    # Skip conversations with fewer than 2 turns
    if len(turns) < 2:
        return samples
    
    # Process each agent response with available context
    for i in range(1, len(turns)):
        current_turn = turns[i]
        
        # Only process agent responses
        if current_turn.get('speaker', '').lower() in ['customer', 'user']:
            continue
        
        response = clean_text(current_turn.get('text', ''))
        
        # Skip empty or very short responses
        if len(response.split()) < 2:
            continue
        
        # Collect context from previous turns
        context_turns = []
        for j in range(max(0, i - max_history), i):
            prev_turn = turns[j]
            speaker = "Customer" if prev_turn.get('speaker', '').lower() in ['customer', 'user'] else "Agent"
            text = clean_text(prev_turn.get('text', ''))
            context_turns.append(f"{speaker}: {text}")
        
        context = "\n".join(context_turns)
        samples.append((context, response))
    
    return samples


def format_for_training(pairs, format_type="instruction"):
    """
    Format pairs for different training approaches
    """
    formatted_samples = []
    
    if format_type == "instruction":
        # Format for instruction fine-tuning (e.g., for Llama, Mistral)
        for instruction, response in pairs:
            formatted_sample = {
                "instruction": instruction,
                "input": "",  # Optional context field, empty for simple instruction-following
                "output": response
            }
            formatted_samples.append(formatted_sample)
    
    elif format_type == "chat":
        # Format for chat fine-tuning (e.g., for chat models)
        for instruction, response in pairs:
            formatted_sample = {
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }
            formatted_samples.append(formatted_sample)
    
    elif format_type == "context":
        # This is for multi-turn context samples
        for context, response in pairs:
            formatted_sample = {
                "instruction": "You are a helpful customer service assistant. Based on the conversation history, provide the next response:",
                "input": context,
                "output": response
            }
            formatted_samples.append(formatted_sample)
    
    return formatted_samples


def process_dataset(dataset, format_types=None, use_multi_turn=False, max_history=3):
    """
    Process the dataset and extract formatted samples
    """
    if format_types is None:
        format_types = ["instruction"]
    
    all_samples = {format_type: [] for format_type in format_types}
    
    for conversation in tqdm(dataset, desc="Processing conversations"):
        # Extract instruction-response pairs
        pairs = extract_instruction_response_pairs(conversation)
        
        # Add formatted samples for each format type
        for format_type in format_types:
            if format_type == "context" and use_multi_turn:
                # For context format, use multi-turn extraction
                context_samples = extract_multi_turn_context(conversation, max_history)
                formatted = format_for_training(context_samples, format_type)
            else:
                # For other formats, use regular pairs
                formatted = format_for_training(pairs, format_type)
            
            all_samples[format_type].extend(formatted)
    
    return all_samples


def create_data_splits(samples, train_size=0.8, val_size=0.1, test_size=0.1, random_seed=42):
    """
    Create train/validation/test splits
    """
    # First split into train and temp (val+test)
    train_samples, temp_samples = train_test_split(
        samples, train_size=train_size, random_state=random_seed
    )
    
    # Then split temp into val and test
    relative_val_size = val_size / (val_size + test_size)
    val_samples, test_samples = train_test_split(
        temp_samples, train_size=relative_val_size, random_state=random_seed
    )
    
    return {
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples
    }


def save_processed_data(data_splits, output_dir, format_type):
    """
    Save processed data to output directory
    """
    format_dir = os.path.join(output_dir, format_type)
    os.makedirs(format_dir, exist_ok=True)
    
    for split_name, samples in data_splits.items():
        output_path = os.path.join(format_dir, f"{split_name}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(samples)} {split_name} samples to {output_path}")


def print_sample(sample, format_type):
    """
    Print a formatted sample
    """
    if format_type == "instruction" or format_type == "context":
        print(f"Instruction: {sample['instruction']}")
        if sample['input']:
            print(f"Input: {sample['input']}")
        print(f"Output: {sample['output']}")
    
    elif format_type == "chat":
        for message in sample['messages']:
            print(f"{message['role'].capitalize()}: {message['content']}")
    
    print("-" * 50)


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Process customer service datasets for chatbot training")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset JSON file")
    parser.add_argument("--output_dir", type=str, default="./training_data", help="Output directory for processed data")
    parser.add_argument("--format", type=str, default="instruction,chat,context", 
                        help="Comma-separated list of format types (instruction, chat, context)")
    parser.add_argument("--multi_turn", action="store_true", help="Use multi-turn context extraction")
    parser.add_argument("--max_history", type=int, default=3, help="Maximum number of turns to include in context")
    parser.add_argument("--train_size", type=float, default=0.8, help="Proportion of data for training")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of data for validation")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate split proportions
    if args.train_size + args.val_size + args.test_size != 1.0:
        print("Warning: Split proportions do not sum to 1.0. Normalizing...")
        total = args.train_size + args.val_size + args.test_size
        args.train_size /= total
        args.val_size /= total
        args.test_size /= total
    
    # Parse format types
    format_types = [fmt.strip() for fmt in args.format.split(",")]
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    dataset = load_dataset(args.input)
    print(f"Loaded {len(dataset)} conversations")
    
    # Process dataset
    print(f"Processing dataset with format types: {format_types}")
    all_samples = process_dataset(
        dataset, 
        format_types=format_types, 
        use_multi_turn=args.multi_turn,
        max_history=args.max_history
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each format type
    for format_type in format_types:
        samples = all_samples[format_type]
        print(f"\nFormat: {format_type}")
        print(f"Extracted {len(samples)} samples")
        
        # Create data splits
        data_splits = create_data_splits(
            samples, 
            train_size=args.train_size, 
            val_size=args.val_size, 
            test_size=args.test_size,
            random_seed=args.seed
        )
        
        # Save processed data
        save_processed_data(data_splits, args.output_dir, format_type)
        
        # Print sample
        if samples:
            print(f"\nSample {format_type} format:")
            print_sample(random.choice(samples), format_type)
    
    print("\nData processing complete!")


if __name__ == "__main__":
    main()