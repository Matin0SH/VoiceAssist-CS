"""
Customer Service Dataset Loader
This script provides functions to download and load common customer service datasets
that don't require API keys.
"""

import os
import json
import zipfile
import tarfile
import pandas as pd
import requests
from tqdm import tqdm
import urllib.request
import xml.etree.ElementTree as ET


def download_file(url, destination):
    """
    Download a file from a URL to a destination with a progress bar
    """
    if os.path.exists(destination):
        print(f"File already exists at {destination}")
        return
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading {url} to {destination}")
    
    # For large files, show a progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as f, tqdm(
            total=total_size, 
            unit='B', 
            unit_scale=True, 
            unit_divisor=1024,
            desc=destination
        ) as progress_bar:
        
        for data in response.iter_content(block_size):
            f.write(data)
            progress_bar.update(len(data))
    
    print(f"Download complete: {destination}")


def load_multiwoz():
    """
    Download and load the MultiWOZ 2.1 dataset
    Returns a list of dialogues
    """
    data_dir = "./data/multiwoz/"
    os.makedirs(data_dir, exist_ok=True)
    
    # Direct link to MultiWOZ 2.1
    url = "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.1.zip"
    zip_path = os.path.join(data_dir, "MultiWOZ_2.1.zip")
    
    # Download if not exists
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    
    # Extract if not already extracted
    data_file = os.path.join(data_dir, "data.json")
    if not os.path.exists(data_file):
        print("Extracting MultiWOZ dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Load the data
    print("Loading MultiWOZ dialogues...")
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to a list of conversations
    conversations = []
    for dialogue_id, dialogue in data.items():
        conversation = {
            'id': dialogue_id,
            'domain': dialogue.get('domain', ''),
            'turns': []
        }
        
        for turn in dialogue.get('log', []):
            conversation['turns'].append({
                'speaker': 'user' if turn.get('metadata', {}) == {} else 'system',
                'text': turn.get('text', ''),
                'dialog_act': turn.get('dialog_act', {}),
                'span_info': turn.get('span_info', [])
            })
        
        conversations.append(conversation)
    
    print(f"Loaded {len(conversations)} conversations from MultiWOZ")
    return conversations


def load_msdialog():
    """
    Download and load the MSDialog dataset
    Returns a DataFrame of conversations
    """
    data_dir = "./data/msdialog/"
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for MSDialog
    urls = {
        "main": "https://github.com/microsoft/MSDialog/raw/master/data/MSDialog-Complete.json",
        "intent": "https://github.com/microsoft/MSDialog/raw/master/data/MSDialog-Intent.xlsx"
    }
    
    main_file = os.path.join(data_dir, "MSDialog-Complete.json")
    intent_file = os.path.join(data_dir, "MSDialog-Intent.xlsx")
    
    # Download files if they don't exist
    if not os.path.exists(main_file):
        download_file(urls["main"], main_file)
    
    if not os.path.exists(intent_file):
        download_file(urls["intent"], intent_file)
    
    # Load the main conversation data
    print("Loading MSDialog conversations...")
    with open(main_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process into a more usable format
    conversations = []
    for thread in data:
        conversation = {
            'id': thread.get('thread_id', ''),
            'title': thread.get('title', ''),
            'category': thread.get('category', ''),
            'turns': []
        }
        
        for utterance in thread.get('utterances', []):
            conversation['turns'].append({
                'id': utterance.get('id', ''),
                'speaker': 'user' if utterance.get('is_answer', False) == False else 'agent',
                'text': utterance.get('utterance', ''),
                'votes': utterance.get('votes', 0),
                'timestamp': utterance.get('timestamp', '')
            })
        
        conversations.append(conversation)
    
    print(f"Loaded {len(conversations)} conversations from MSDialog")
    
    # Optionally load the intent annotations
    try:
        intent_data = pd.read_excel(intent_file)
        print(f"Loaded intent annotations with {len(intent_data)} entries")
        # You can merge this with the conversation data if needed
    except Exception as e:
        print(f"Could not load intent annotations: {e}")
    
    return conversations


def load_ubuntu_corpus():
    """
    Download and load the Ubuntu Dialogue Corpus
    Returns a DataFrame of the training data (as it's the most manageable portion)
    """
    data_dir = "./data/ubuntu_corpus/"
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for the Ubuntu Dialogue Corpus (training set)
    url = "https://www.dropbox.com/s/2fdn26rj6h9bpvl/ubuntu_data.zip"
    zip_path = os.path.join(data_dir, "ubuntu_data.zip")
    
    # The Ubuntu dataset is large, so let's confirm before downloading
    if not os.path.exists(zip_path):
        print(f"The Ubuntu Dialogue Corpus is approximately 1.7GB compressed.")
        proceed = input("Do you want to download it? (y/n): ")
        
        if proceed.lower() == 'y':
            download_file(url, zip_path)
        else:
            print("Skipping Ubuntu Dialogue Corpus download")
            return None
    
    # Extract if not already extracted
    train_file = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_file):
        print("Extracting Ubuntu Dialogue Corpus (this may take a while)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Load the training data
    print("Loading Ubuntu Dialogue Corpus (training set)...")
    try:
        # This dataset is large, so we'll use pandas to read it efficiently
        train_data = pd.read_csv(train_file)
        print(f"Loaded {len(train_data)} dialogue examples from Ubuntu Corpus training set")
        return train_data
    except Exception as e:
        print(f"Error loading Ubuntu Corpus: {e}")
        return None


def load_taskmaster():
    """
    Download and load the Taskmaster-1 dataset
    Returns a list of dialogues
    """
    data_dir = "./data/taskmaster/"
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for Taskmaster-1
    url = "https://storage.googleapis.com/dialog-data-corpus/TASKMASTER-1-2019/self-dialogs.json"
    json_path = os.path.join(data_dir, "self-dialogs.json")
    
    # Download if not exists
    if not os.path.exists(json_path):
        download_file(url, json_path)
    
    # Load the data
    print("Loading Taskmaster-1 dialogues...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} conversations from Taskmaster-1")
    return data


def load_customer_support_on_twitter(sample_size=10000):
    """
    Download a sample of the Customer Support on Twitter dataset
    The full dataset is too large, so we'll download a sample
    Returns a DataFrame of tweets
    """
    data_dir = "./data/twitter_customer_support/"
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for Customer Support on Twitter dataset
    url = "https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter/download?datasetVersionNumber=9"
    
    print("Note: The Customer Support on Twitter dataset requires a Kaggle account.")
    print("Please download it manually from:")
    print("https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
    print("After downloading, place the 'twcs.csv' file in the following directory:")
    print(f"{os.path.abspath(data_dir)}")
    
    csv_path = os.path.join(data_dir, "twcs.csv")
    
    # Check if the file exists
    if os.path.exists(csv_path):
        print(f"Loading {sample_size} samples from Customer Support on Twitter dataset...")
        # This dataset is very large, so we'll use pandas with nrows to limit memory usage
        data = pd.read_csv(csv_path, nrows=sample_size)
        print(f"Loaded {len(data)} tweets from Customer Support on Twitter")
        return data
    else:
        print("File not found. Please download the dataset manually.")
        return None


def load_schema_guided_dialogue():
    """
    Download and load the Schema-Guided Dialogue dataset
    Returns a dictionary of splits (train, dev, test) with their dialogues
    """
    data_dir = "./data/schema_guided_dialogue/"
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for Schema-Guided Dialogue dataset
    url = "https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/archive/refs/heads/master.zip"
    zip_path = os.path.join(data_dir, "dstc8-schema-guided-dialogue-master.zip")
    
    # Download if not exists
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    
    # Extract if not already extracted
    extracted_dir = os.path.join(data_dir, "dstc8-schema-guided-dialogue-master")
    if not os.path.exists(extracted_dir):
        print("Extracting Schema-Guided Dialogue dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Load the data (focusing on train split for example)
    print("Loading Schema-Guided Dialogue dataset (train split)...")
    
    result = {}
    for split in ['train', 'dev', 'test']:
        split_dir = os.path.join(extracted_dir, split)
        if not os.path.exists(split_dir):
            print(f"Split directory not found: {split_dir}")
            continue
        
        dialogues = []
        # The dataset is split across multiple files
        for filename in os.listdir(split_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(split_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    dialogues.extend(data)
        
        result[split] = dialogues
        print(f"Loaded {len(dialogues)} dialogues from {split} split")
    
    return result


def convert_to_standard_format(dataset, dataset_name):
    """
    Convert various datasets to a standardized format for easier processing
    Returns a list of standardized conversations
    """
    standardized = []
    
    if dataset_name == "multiwoz":
        # MultiWOZ is already in a good format from our load function
        return dataset
    
    elif dataset_name == "msdialog":
        # MSDialog format conversion
        for conv in dataset:
            standardized_conv = {
                "id": conv["id"],
                "source": "msdialog",
                "metadata": {
                    "title": conv["title"],
                    "category": conv["category"]
                },
                "turns": []
            }
            
            for turn in conv["turns"]:
                standardized_conv["turns"].append({
                    "speaker": "customer" if turn["speaker"] == "user" else "agent",
                    "text": turn["text"],
                    "metadata": {
                        "id": turn["id"],
                        "votes": turn["votes"],
                        "timestamp": turn["timestamp"]
                    }
                })
            
            standardized.append(standardized_conv)
    
    elif dataset_name == "taskmaster":
        # Taskmaster format conversion
        for conv in dataset:
            standardized_conv = {
                "id": conv.get("conversation_id", ""),
                "source": "taskmaster",
                "metadata": {
                    "scenario": conv.get("instruction_id", ""),
                    "domain": conv.get("conversation_domain", "")
                },
                "turns": []
            }
            
            for turn in conv.get("utterances", []):
                standardized_conv["turns"].append({
                    "speaker": "customer" if turn.get("speaker") == "USER" else "agent",
                    "text": turn.get("text", ""),
                    "metadata": {
                        "segment_id": turn.get("segment_id", "")
                    }
                })
            
            standardized.append(standardized_conv)
    
    elif dataset_name == "schema_guided":
        # Schema-Guided Dialogue format conversion
        for split, dialogues in dataset.items():
            for dialog in dialogues:
                standardized_conv = {
                    "id": dialog.get("dialogue_id", ""),
                    "source": f"schema_guided_{split}",
                    "metadata": {
                        "services": dialog.get("services", [])
                    },
                    "turns": []
                }
                
                for turn in dialog.get("turns", []):
                    standardized_conv["turns"].append({
                        "speaker": "customer" if turn.get("speaker") == "USER" else "agent",
                        "text": turn.get("utterance", ""),
                        "metadata": {
                            "frames": turn.get("frames", [])
                        }
                    })
                
                standardized.append(standardized_conv)
    
    else:
        print(f"Conversion for {dataset_name} not implemented")
        return dataset
    
    return standardized


def save_dataset(data, name, format="json"):
    """
    Save a dataset to disk
    """
    output_dir = "./processed_data/"
    os.makedirs(output_dir, exist_ok=True)
    
    if format == "json":
        output_path = os.path.join(output_dir, f"{name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    elif format == "csv":
        output_path = os.path.join(output_dir, f"{name}.csv")
        pd.DataFrame(data).to_csv(output_path, index=False)
    
    print(f"Saved {name} dataset to {output_path}")


def main():
    """
    Main function to demonstrate dataset loading
    """
    print("Customer Service Dataset Loader")
    print("===============================")
    
    datasets = {
        "1": {"name": "MultiWOZ", "loader": load_multiwoz},
        "2": {"name": "MSDialog", "loader": load_msdialog},
        "3": {"name": "Taskmaster", "loader": load_taskmaster},
        "4": {"name": "Schema-Guided Dialogue", "loader": load_schema_guided_dialogue},
        # Ubuntu corpus is very large, so let's not include it in the default options
        # "5": {"name": "Ubuntu Dialogue Corpus", "loader": load_ubuntu_corpus},
    }
    
    # Print available datasets
    print("\nAvailable datasets:")
    for key, dataset in datasets.items():
        print(f"{key}. {dataset['name']}")
    
    # Ask user which datasets to load
    choices = input("\nEnter dataset numbers to load (comma-separated, or 'all'): ")
    
    if choices.lower() == 'all':
        choices = list(datasets.keys())
    else:
        choices = [choice.strip() for choice in choices.split(',')]
    
    # Load selected datasets
    loaded_datasets = {}
    for choice in choices:
        if choice in datasets:
            print(f"\nLoading {datasets[choice]['name']}...")
            try:
                data = datasets[choice]["loader"]()
                if data is not None:
                    # Convert to standard format
                    standardized_data = convert_to_standard_format(
                        data, datasets[choice]["name"].lower().replace("-", "_")
                    )
                    loaded_datasets[datasets[choice]["name"]] = standardized_data
                    
                    # Save the standardized data
                    save_dataset(
                        standardized_data, 
                        datasets[choice]["name"].lower().replace(" ", "_").replace("-", "_")
                    )
            except Exception as e:
                print(f"Error loading {datasets[choice]['name']}: {e}")
        else:
            print(f"Invalid choice: {choice}")
    
    print("\nSummary:")
    for name, data in loaded_datasets.items():
        print(f"- {name}: {len(data)} conversations")
    
    print("\nAll processed datasets have been saved to the 'processed_data' directory.")


if __name__ == "__main__":
    main()