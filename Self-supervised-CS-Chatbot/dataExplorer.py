# """
# Customer Service Dataset Explorer
# This script helps explore and visualize the loaded datasets
# """

# import os
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter, def generate_dataset_report(dataset, dataset_name):
#     """
#     Generate a comprehensive report about the dataset
#     """
#     # Create output directory
#     output_dir = "./reports/"
#     os.makedirs(output_dir, exist_ok=True)
#     report_path = os.path.join(output_dir, f"{dataset_name}_report.txt")
    
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write(f"DATASET REPORT: {dataset_name}\n")
#         f.write("=" * 50 + "\n\n")
        
#         # Basic statistics
#         f.write("BASIC STATISTICS\n")
#         f.write("-" * 30 + "\n")
#         stats = get_basic_stats(dataset)
#         for key, value in stats.items():
#             f.write(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}\n")
#         f.write("\n")
        
#         # Vocabulary analysis
#         f.write("VOCABULARY ANALYSIS\n")
#         f.write("-" * 30 + "\n")
#         vocab_stats = analyze_vocabulary(dataset)
#         f.write(f"Customer vocabulary size: {vocab_stats['customer_vocab_size']}\n")
#         f.write(f"Agent vocabulary size: {vocab_stats['agent_vocab_size']}\n")
        
#         f.write("\nMost common customer words:\n")
#         for word, count in vocab_stats['most_common_customer']:
#             f.write(f"- {word}: {count}\n")
        
#         f.write("\nMost common agent words:\n")
#         for word, count in vocab_stats['most_common_agent']:
#             f.write(f"- {word}: {count}\n")
#         f.write("\n")
        
#         # Conversation patterns
#         f.write("CONVERSATION PATTERNS\n")
#         f.write("-" * 30 + "\n")
#         pattern_stats = analyze_conversation_patterns(dataset)
        
#         f.write("Conversation initiator distribution:\n")
#         for initiator, count in pattern_stats['initiator_distribution'].items():
#             f.write(f"- {initiator}: {count}\n")
        
#         f.write(f"\nFollow-up question ratio: {pattern_stats['follow_up_question_ratio']:.2f}\n")
#         f.write(f"Average agent/customer response length ratio: {pattern_stats['avg_response_length_ratio']:.2f}\n")
#         f.write("\n")
        
#         # Common intents
#         f.write("COMMON INTENT PHRASES\n")
#         f.write("-" * 30 + "\n")
#         common_intents = extract_common_intents(dataset)
#         for intent, count in common_intents:
#             f.write(f"- {intent}: {count}\n")
#         f.write("\n")
        
#         # Sample conversations
#         f.write("SAMPLE CONVERSATIONS\n")
#         f.write("-" * 30 + "\n")
        
#         import random
#         sample_indices = random.sample(range(len(dataset)), min(3, len(dataset)))
        
#         for i, idx in enumerate(sample_indices):
#             conv = dataset[idx]
#             f.write(f"\nSample Conversation {i+1} (ID: {conv.get('id', 'N/A')}):\n")
            
#             for turn in conv.get("turns", []):
#                 speaker = turn.get("speaker", "").upper()
#                 text = turn.get("text", "")
#                 f.write(f"{speaker}: {text}\n")
            
#             f.write("\n")
    
#     print(f"Report generated at {report_path}")
#     return report_path


# def main():
#     """
#     Main function to explore datasets
#     """
#     # Check for processed datasets
#     processed_dir = "./processed_data/"
#     if not os.path.exists(processed_dir):
#         print("No processed datasets found. Please run the dataset loader first.")
#         return
    
#     # Get available datasets
#     dataset_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
    
#     if not dataset_files:
#         print("No processed datasets found. Please run the dataset loader first.")
#         return
    
#     print("Available datasets:")
#     for i, dataset_file in enumerate(dataset_files):
#         print(f"{i+1}. {dataset_file[:-5]}")  # Remove .json extension
    
#     # Ask user which datasets to explore
#     choices = input("\nEnter dataset numbers to explore (comma-separated, or 'all'): ")
    
#     if choices.lower() == 'all':
#         selected_indices = range(len(dataset_files))
#     else:
#         selected_indices = [int(choice.strip()) - 1 for choice in choices.split(',')]
    
#     # Load and explore selected datasets
#     loaded_datasets = {}
#     for idx in selected_indices:
#         if 0 <= idx < len(dataset_files):
#             dataset_name = dataset_files[idx][:-5]  # Remove .json extension
#             print(f"\nExploring {dataset_name}...")
            
#             dataset = load_saved_dataset(dataset_name)
#             if dataset:
#                 loaded_datasets[dataset_name] = dataset
                
#                 # Generate basic statistics
#                 stats = get_basic_stats(dataset)
#                 print("\nBasic Statistics:")
#                 for key, value in stats.items():
#                     print(f"- {key}: {value:.2f}" if isinstance(value, float) else f"- {key}: {value}")
                
#                 # Create visualizations
#                 print("\nGenerating visualizations...")
#                 plot_conversation_length_distribution(dataset, dataset_name)
#                 plot_utterance_length_distribution(dataset, dataset_name)
#                 plot_common_intents(dataset, dataset_name)
                
#                 # Generate report
#                 report_path = generate_dataset_report(dataset, dataset_name)
#                 print(f"Report generated at {report_path}")
                
#                 # Print sample conversations
#                 print_sample_conversations(dataset, 1)
#         else:
#             print(f"Invalid index: {idx+1}")
    
#     # Ask if user wants to combine datasets
#     if len(loaded_datasets) > 1:
#         combine = input("\nDo you want to combine all explored datasets? (y/n): ")
#         if combine.lower() == 'y':
#             combined_dataset = combine_datasets(loaded_datasets)
#             print(f"\nCombined dataset has {len(combined_dataset)} conversations")
            
#             # Save combined dataset
#             combined_name = "combined_dataset"
#             save_path = os.path.join(processed_dir, f"{combined_name}.json")
#             with open(save_path, 'w', encoding='utf-8') as f:
#                 json.dump(combined_dataset, f, ensure_ascii=False, indent=2)
            
#             print(f"Combined dataset saved to {save_path}")
            
#             # Generate statistics for combined dataset
#             stats = get_basic_stats(combined_dataset)
#             print("\nBasic Statistics for Combined Dataset:")
#             for key, value in stats.items():
#                 print(f"- {key}: {value:.2f}" if isinstance(value, float) else f"- {key}: {value}")
            
#             # Generate report for combined dataset
#             report_path = generate_dataset_report(combined_dataset, combined_name)


# if __name__ == "__main__":
#     main()
# aultdict
# import nltk
# from nltk.tokenize import word_tokenize
# import numpy as np

# # Download NLTK data if needed
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')


# def load_saved_dataset(name):
#     """
#     Load a dataset saved by the dataset loader
#     """
#     dataset_path = f"./processed_data/{name}.json"
#     if os.path.exists(dataset_path):
#         with open(dataset_path, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     else:
#         print(f"Dataset {name} not found at {dataset_path}")
#         return None


# def get_basic_stats(dataset):
#     """
#     Calculate basic statistics about a dataset
#     """
#     total_conversations = len(dataset)
    
#     turn_counts = []
#     utterance_lengths = []
#     customer_utterance_lengths = []
#     agent_utterance_lengths = []
    
#     for conv in dataset:
#         turns = conv.get("turns", [])
#         turn_counts.append(len(turns))
        
#         for turn in turns:
#             text = turn.get("text", "")
#             words = word_tokenize(text)
#             utterance_lengths.append(len(words))
            
#             if turn.get("speaker", "").lower() in ["customer", "user"]:
#                 customer_utterance_lengths.append(len(words))
#             else:
#                 agent_utterance_lengths.append(len(words))
    
#     stats = {
#         "total_conversations": total_conversations,
#         "avg_turns_per_conversation": np.mean(turn_counts) if turn_counts else 0,
#         "min_turns": min(turn_counts) if turn_counts else 0,
#         "max_turns": max(turn_counts) if turn_counts else 0,
#         "total_utterances": sum(turn_counts) if turn_counts else 0,
#         "avg_utterance_length": np.mean(utterance_lengths) if utterance_lengths else 0,
#         "avg_customer_utterance_length": np.mean(customer_utterance_lengths) if customer_utterance_lengths else 0,
#         "avg_agent_utterance_length": np.mean(agent_utterance_lengths) if agent_utterance_lengths else 0,
#     }
    
#     return stats


# def analyze_vocabulary(dataset):
#     """
#     Analyze vocabulary used in the dataset
#     """
#     customer_words = []
#     agent_words = []
    
#     for conv in dataset:
#         for turn in conv.get("turns", []):
#             text = turn.get("text", "").lower()
#             words = word_tokenize(text)
            
#             if turn.get("speaker", "").lower() in ["customer", "user"]:
#                 customer_words.extend(words)
#             else:
#                 agent_words.extend(words)
    
#     customer_word_freq = Counter(customer_words)
#     agent_word_freq = Counter(agent_words)
    
#     # Calculate vocabulary size
#     customer_vocab_size = len(customer_word_freq)
#     agent_vocab_size = len(agent_word_freq)
    
#     # Get most common words
#     most_common_customer = customer_word_freq.most_common(20)
#     most_common_agent = agent_word_freq.most_common(20)
    
#     return {
#         "customer_vocab_size": customer_vocab_size,
#         "agent_vocab_size": agent_vocab_size,
#         "most_common_customer": most_common_customer,
#         "most_common_agent": most_common_agent,
#     }


# def analyze_conversation_patterns(dataset):
#     """
#     Analyze conversation patterns
#     """
#     # Analyze who initiates conversations
#     initiators = []
    
#     # Check for turns that reference previous turns
#     follow_up_questions = 0
#     total_customer_turns = 0
    
#     # Analyze conversation flow patterns
#     avg_response_lengths = []
    
#     for conv in dataset:
#         turns = conv.get("turns", [])
#         if turns:
#             initiators.append(turns[0].get("speaker", ""))
        
#         customer_texts = []
#         agent_texts = []
        
#         for i, turn in enumerate(turns):
#             speaker = turn.get("speaker", "").lower()
#             text = turn.get("text", "")
            
#             if speaker in ["customer", "user"]:
#                 total_customer_turns += 1
#                 customer_texts.append(text)
                
#                 # Heuristic for follow-up questions: check if question mark and 
#                 # it's not the first customer turn in conversation
#                 if "?" in text and len(customer_texts) > 1:
#                     follow_up_questions += 1
#             else:
#                 agent_texts.append(text)
                
#             # Calculate response length ratio if we have a customer-agent pair
#             if i > 0 and speaker not in ["customer", "user"] and turns[i-1].get("speaker", "").lower() in ["customer", "user"]:
#                 customer_words = len(word_tokenize(turns[i-1].get("text", "")))
#                 agent_words = len(word_tokenize(text))
#                 if customer_words > 0:  # Avoid division by zero
#                     avg_response_lengths.append(agent_words / customer_words)
    
#     # Calculate initiator distribution
#     initiator_counts = Counter(initiators)
    
#     return {
#         "initiator_distribution": dict(initiator_counts),
#         "follow_up_question_ratio": follow_up_questions / total_customer_turns if total_customer_turns > 0 else 0,
#         "avg_response_length_ratio": np.mean(avg_response_lengths) if avg_response_lengths else 0,
#     }


# def extract_common_intents(dataset, num_intents=10):
#     """
#     Try to extract common intents (very simple approach)
#     """
#     intent_phrases = [
#         "how do i", "how can i", "can you", "i need to", "i want to",
#         "help me", "is there", "what is", "why is", "when will",
#         "problem with", "not working", "doesn't work", "error", "issue",
#         "thank you", "thanks", "appreciate", "grateful", "update",
#         "login", "password", "account", "reset", "change",
#         "cancel", "refund", "return", "upgrade", "downgrade"
#     ]
    
#     intent_counts = defaultdict(int)
    
#     for conv in dataset:
#         for turn in conv.get("turns", []):
#             if turn.get("speaker", "").lower() in ["customer", "user"]:
#                 text = turn.get("text", "").lower()
                
#                 for phrase in intent_phrases:
#                     if phrase in text:
#                         intent_counts[phrase] += 1
    
#     # Get most common intents
#     most_common_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:num_intents]
    
#     return most_common_intents


# def plot_conversation_length_distribution(dataset, dataset_name):
#     """
#     Plot distribution of conversation lengths
#     """
#     turn_counts = [len(conv.get("turns", [])) for conv in dataset]
    
#     plt.figure(figsize=(10, 6))
#     sns.histplot(turn_counts, kde=True, bins=20)
#     plt.title(f"Conversation Length Distribution - {dataset_name}")
#     plt.xlabel("Number of Turns")
#     plt.ylabel("Count")
#     plt.tight_layout()
    
#     # Save the plot
#     output_dir = "./visualizations/"
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f"{dataset_name}_conversation_length.png"))
#     plt.close()


# def plot_utterance_length_distribution(dataset, dataset_name):
#     """
#     Plot distribution of utterance lengths for customers and agents
#     """
#     customer_lengths = []
#     agent_lengths = []
    
#     for conv in dataset:
#         for turn in conv.get("turns", []):
#             text = turn.get("text", "")
#             words = word_tokenize(text)
            
#             if turn.get("speaker", "").lower() in ["customer", "user"]:
#                 customer_lengths.append(len(words))
#             else:
#                 agent_lengths.append(len(words))
    
#     plt.figure(figsize=(12, 6))
    
#     plt.subplot(1, 2, 1)
#     sns.histplot(customer_lengths, kde=True, bins=20)
#     plt.title(f"Customer Utterance Lengths - {dataset_name}")
#     plt.xlabel("Number of Words")
#     plt.ylabel("Count")
#     plt.xlim(0, min(100, max(customer_lengths)))  # Limit x-axis for better visualization
    
#     plt.subplot(1, 2, 2)
#     sns.histplot(agent_lengths, kde=True, bins=20)
#     plt.title(f"Agent Utterance Lengths - {dataset_name}")
#     plt.xlabel("Number of Words")
#     plt.ylabel("Count")
#     plt.xlim(0, min(100, max(agent_lengths)))  # Limit x-axis for better visualization
    
#     plt.tight_layout()
    
#     # Save the plot
#     output_dir = "./visualizations/"
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f"{dataset_name}_utterance_length.png"))
#     plt.close()


# def plot_common_intents(dataset, dataset_name):
#     """
#     Plot common intents
#     """
#     common_intents = extract_common_intents(dataset)
    
#     intent_labels = [intent for intent, count in common_intents]
#     intent_counts = [count for intent, count in common_intents]
    
#     plt.figure(figsize=(12, 6))
#     bars = plt.barh(intent_labels, intent_counts)
    
#     # Add values to bars
#     for bar in bars:
#         width = bar.get_width()
#         plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.0f}", 
#                  ha='left', va='center')
    
#     plt.title(f"Common Intent Phrases - {dataset_name}")
#     plt.xlabel("Count")
#     plt.tight_layout()
    
#     # Save the plot
#     output_dir = "./visualizations/"
#     os.makedirs(output_dir, exist_ok=True)
#     plt.savefig(os.path.join(output_dir, f"{dataset_name}_common_intents.png"))
#     plt.close()


# def print_sample_conversations(dataset, num_samples=2):
#     """
#     Print sample conversations from the dataset
#     """
#     import random
    
#     sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
#     for i, idx in enumerate(sample_indices):
#         conv = dataset[idx]
#         print(f"\nSample Conversation {i+1} (ID: {conv.get('id', 'N/A')}):")
#         print("-" * 50)
        
#         for turn in conv.get("turns", []):
#             speaker = turn.get("speaker", "").upper()
#             text = turn.get("text", "")
#             print(f"{speaker}: {text}")
        
#         print("-" * 50)


# def combine_datasets(datasets):
#     """
#     Combine multiple datasets into one
#     """
#     combined = []
    
#     for name, dataset in datasets.items():
#         for conv in dataset:
#             # Add dataset source to metadata
#             if "metadata" not in conv:
#                 conv["metadata"] = {}
#             conv["metadata"]["original_dataset"] = name
#             combined.append(conv)
    
#     return combined


# def


"""
MultiWOZ Dataset Explorer
This script explores and analyzes the MultiWOZ dataset structure and content
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import glob
from tqdm import tqdm

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('punkt_tab')


def load_multiwoz_dataset(data_dir="./data/multiwoz"):
    """
    Load all JSON files from the MultiWOZ dataset directory
    Returns a dictionary with filename as key and content as value
    """
    dataset_files = {}
    
    # Get all JSON files in the directory
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return None
    
    print(f"Found {len(json_files)} JSON files in the directory")
    
    # Load each JSON file
    for file_path in tqdm(json_files, desc="Loading JSON files"):
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                dataset_files[filename] = data
                print(f"Loaded {filename} with {len(data) if isinstance(data, list) or isinstance(data, dict) else 'N/A'} entries")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return dataset_files


def analyze_data_json(data, max_dialogs=5):
    """
    Analyze the main data.json file which contains the dialogues
    """
    print("\nAnalyzing data.json...")
    
    # Check if data is a dictionary
    if not isinstance(data, dict):
        print("data.json is not in the expected format (should be a dictionary)")
        return
    
    num_dialogues = len(data)
    print(f"Number of dialogues: {num_dialogues}")
    
    # Sample a few dialogues to understand structure
    print("\nSample dialogue structure:")
    sampled_ids = list(data.keys())[:max_dialogs]
    
    for dialogue_id in sampled_ids:
        dialogue = data[dialogue_id]
        print(f"\nDialogue ID: {dialogue_id}")
        print(f"Keys in dialogue: {list(dialogue.keys())}")
        
        # Check if 'goal' exists
        if 'goal' in dialogue:
            print(f"Goal domains: {list(dialogue['goal'].keys()) if isinstance(dialogue['goal'], dict) else 'Not a dictionary'}")
        
        # Check if 'log' exists and show sample turns
        if 'log' in dialogue:
            num_turns = len(dialogue['log'])
            print(f"Number of turns: {num_turns}")
            
            # Show first few turns
            for i, turn in enumerate(dialogue['log'][:2]):
                print(f"  Turn {i+1} keys: {list(turn.keys())}")
                print(f"  Turn {i+1} text: {turn.get('text', 'No text')}")
                print(f"  Turn {i+1} metadata: {'Present' if turn.get('metadata') else 'None'}")
            
            if num_turns > 2:
                print(f"  ... {num_turns-2} more turns ...")
    
    # Collect statistics about dialogue length
    turn_counts = [len(dialogue.get('log', [])) for dialogue in data.values()]
    
    print("\nDialogue length statistics:")
    print(f"  Average number of turns: {np.mean(turn_counts):.2f}")
    print(f"  Min number of turns: {min(turn_counts) if turn_counts else 0}")
    print(f"  Max number of turns: {max(turn_counts) if turn_counts else 0}")
    
    return {
        "num_dialogues": num_dialogues,
        "turn_counts": turn_counts
    }


def analyze_domain_files(files_dict):
    """
    Analyze domain-specific files like restaurant_db.json, hotel_db.json, etc.
    """
    print("\nAnalyzing domain-specific files...")
    
    domain_files = {k: v for k, v in files_dict.items() if k.endswith('_db.json')}
    
    if not domain_files:
        print("No domain-specific files found")
        return
    
    for filename, data in domain_files.items():
        print(f"\nFile: {filename}")
        
        if not isinstance(data, list):
            print(f"  Unexpected format: {type(data)}")
            continue
        
        print(f"  Number of entries: {len(data)}")
        
        if data:
            # Get a sample entry to show structure
            sample = data[0]
            print(f"  Sample entry keys: {list(sample.keys())}")
            
            # Count occurrences of each key across all entries
            key_counts = Counter()
            for entry in data:
                key_counts.update(entry.keys())
            
            print("  Key frequency across entries:")
            for key, count in key_counts.most_common():
                print(f"    {key}: {count}/{len(data)} entries ({count/len(data)*100:.1f}%)")


def analyze_ontology(files_dict):
    """
    Analyze the ontology.json file
    """
    print("\nAnalyzing ontology.json...")
    
    if 'ontology.json' not in files_dict:
        print("ontology.json not found")
        return
    
    ontology = files_dict['ontology.json']
    
    if not isinstance(ontology, dict):
        print(f"Unexpected format: {type(ontology)}")
        return
    
    print(f"Number of keys in ontology: {len(ontology)}")
    
    # Count different types of slots
    inform_count = sum(1 for k in ontology.keys() if 'inform' in k.lower())
    request_count = sum(1 for k in ontology.keys() if 'request' in k.lower())
    book_count = sum(1 for k in ontology.keys() if 'book' in k.lower())
    
    print(f"Inform slots: {inform_count}")
    print(f"Request slots: {request_count}")
    print(f"Booking slots: {book_count}")
    
    # Sample a few values
    sample_keys = list(ontology.keys())[:5]
    print("\nSample ontology entries:")
    for key in sample_keys:
        values = ontology[key]
        print(f"  {key}: {len(values)} values")
        print(f"    Sample values: {values[:3]}...")


def extract_dialogue_acts(files_dict):
    """
    Extract and analyze the dialogue acts from system_acts.json or similar
    """
    print("\nAnalyzing dialogue acts...")
    
    act_files = [f for f in files_dict.keys() if 'act' in f.lower()]
    
    if not act_files:
        print("No dialogue act files found")
        return
    
    for filename in act_files:
        acts_data = files_dict[filename]
        print(f"\nFile: {filename}")
        
        if isinstance(acts_data, dict):
            print(f"Number of entries: {len(acts_data)}")
            
            # Count act types
            act_types = Counter()
            for dialogue_id, acts in acts_data.items():
                if isinstance(acts, list):
                    act_types.update(acts)
                elif isinstance(acts, dict):
                    act_types.update(acts.keys())
            
            print("\nMost common dialogue acts:")
            for act, count in act_types.most_common(10):
                print(f"  {act}: {count}")
        else:
            print(f"Unexpected format: {type(acts_data)}")


def extract_conversations(data_json):
    """
    Extract conversations from data.json into a more usable format
    Returns a list of conversations
    """
    conversations = []
    
    if not isinstance(data_json, dict):
        return conversations
    
    for dialogue_id, dialogue in tqdm(data_json.items(), desc="Extracting conversations"):
        conversation = {
            'id': dialogue_id,
            'domain': dialogue.get('domain', 'unknown'),
            'turns': []
        }
        
        for turn in dialogue.get('log', []):
            # Determine speaker (user turns have empty metadata)
            speaker = 'user' if turn.get('metadata', {}) == {} else 'system'
            
            conversation['turns'].append({
                'speaker': speaker,
                'text': turn.get('text', ''),
                'dialog_act': turn.get('dialog_act', {}),
                'span_info': turn.get('span_info', [])
            })
        
        conversations.append(conversation)
    
    return conversations


def analyze_conversations(conversations):
    """
    Analyze extracted conversations
    """
    print(f"\nAnalyzing {len(conversations)} conversations...")
    
    # Collect statistics
    user_turns = 0
    system_turns = 0
    user_utterance_lengths = []
    system_utterance_lengths = []
    
    domains = Counter()
    
    for conv in conversations:
        domains[conv.get('domain', 'unknown')] += 1
        
        for turn in conv.get('turns', []):
            text = turn.get('text', '')
            words = word_tokenize(text)
            
            if turn.get('speaker') == 'user':
                user_turns += 1
                user_utterance_lengths.append(len(words))
            else:
                system_turns += 1
                system_utterance_lengths.append(len(words))
    
    print(f"Total turns: {user_turns + system_turns}")
    print(f"User turns: {user_turns}")
    print(f"System turns: {system_turns}")
    
    print("\nAverage utterance length (words):")
    print(f"  User: {np.mean(user_utterance_lengths):.2f}")
    print(f"  System: {np.mean(system_utterance_lengths):.2f}")
    
    print("\nDomain distribution:")
    for domain, count in domains.most_common():
        print(f"  {domain}: {count} conversations ({count/len(conversations)*100:.1f}%)")
    
    # Find common user requests
    user_texts = [turn.get('text', '').lower() for conv in conversations for turn in conv.get('turns', []) if turn.get('speaker') == 'user']
    
    # Look for common questions or requests
    print("\nAnalyzing common user questions/requests...")
    patterns = ["where", "how", "what", "when", "which", "can you", "is there", "are there", "do you", "i want", "i need", "looking for", "book"]
    
    pattern_counts = {pattern: sum(1 for text in user_texts if pattern in text.split()) for pattern in patterns}
    
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{pattern}': {count} occurrences ({count/len(user_texts)*100:.1f}% of user turns)")
    
    return {
        "user_utterance_lengths": user_utterance_lengths,
        "system_utterance_lengths": system_utterance_lengths,
        "domains": domains
    }


def plot_distributions(stats):
    """
    Create visualizations for the dataset
    """
    print("\nGenerating visualizations...")
    output_dir = "./visualizations/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot turn count distribution
    if "turn_counts" in stats:
        plt.figure(figsize=(10, 6))
        sns.histplot(stats["turn_counts"], kde=True, bins=20)
        plt.title("Conversation Length Distribution")
        plt.xlabel("Number of Turns")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multiwoz_turn_counts.png"))
        plt.close()
        print(f"Saved turn count distribution to {os.path.join(output_dir, 'multiwoz_turn_counts.png')}")
    
    # Plot utterance length distributions
    if "user_utterance_lengths" in stats and "system_utterance_lengths" in stats:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(stats["user_utterance_lengths"], kde=True, bins=20)
        plt.title("User Utterance Lengths")
        plt.xlabel("Number of Words")
        plt.ylabel("Count")
        plt.xlim(0, min(50, max(stats["user_utterance_lengths"])))
        
        plt.subplot(1, 2, 2)
        sns.histplot(stats["system_utterance_lengths"], kde=True, bins=20)
        plt.title("System Utterance Lengths")
        plt.xlabel("Number of Words")
        plt.ylabel("Count")
        plt.xlim(0, min(50, max(stats["system_utterance_lengths"])))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multiwoz_utterance_lengths.png"))
        plt.close()
        print(f"Saved utterance length distributions to {os.path.join(output_dir, 'multiwoz_utterance_lengths.png')}")
    
    # Plot domain distribution
    if "domains" in stats:
        domains = stats["domains"]
        plt.figure(figsize=(12, 6))
        domain_names = [domain for domain, _ in domains.most_common()]
        domain_counts = [count for _, count in domains.most_common()]
        
        bars = plt.barh(domain_names, domain_counts)
        
        # Add counts to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.0f}", 
                     ha='left', va='center')
        
        plt.title("Domain Distribution")
        plt.xlabel("Number of Conversations")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multiwoz_domains.png"))
        plt.close()
        print(f"Saved domain distribution to {os.path.join(output_dir, 'multiwoz_domains.png')}")


def print_sample_conversations(conversations, num_samples=2):
    """
    Print sample conversations
    """
    print(f"\nSample Conversations (showing {num_samples}):")
    
    import random
    sample_indices = random.sample(range(len(conversations)), min(num_samples, len(conversations)))
    
    for i, idx in enumerate(sample_indices):
        conv = conversations[idx]
        print(f"\nSample Conversation {i+1} (ID: {conv.get('id', 'unknown')}, Domain: {conv.get('domain', 'unknown')}):")
        print("-" * 50)
        
        for turn in conv.get('turns', []):
            speaker = turn.get('speaker', '').upper()
            text = turn.get('text', '')
            print(f"{speaker}: {text}")
        
        print("-" * 50)


def generate_report(stats, conversations):
    """
    Generate a comprehensive report about the dataset
    """
    output_dir = "./reports/"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "multiwoz_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("MULTIWOZ DATASET REPORT\n")
        f.write("======================\n\n")
        
        f.write("BASIC STATISTICS\n")
        f.write("--------------\n")
        f.write(f"Number of dialogues: {stats.get('num_dialogues', 'N/A')}\n")
        
        turn_counts = stats.get('turn_counts', [])
        if turn_counts:
            f.write(f"Average turns per dialogue: {np.mean(turn_counts):.2f}\n")
            f.write(f"Min turns: {min(turn_counts)}\n")
            f.write(f"Max turns: {max(turn_counts)}\n")
        
        f.write("\nUTTERANCE STATISTICS\n")
        f.write("------------------\n")
        
        user_lengths = stats.get('user_utterance_lengths', [])
        system_lengths = stats.get('system_utterance_lengths', [])
        
        if user_lengths:
            f.write(f"User utterances - Average length: {np.mean(user_lengths):.2f} words\n")
            f.write(f"User utterances - Min length: {min(user_lengths)} words\n")
            f.write(f"User utterances - Max length: {max(user_lengths)} words\n")
        
        if system_lengths:
            f.write(f"System utterances - Average length: {np.mean(system_lengths):.2f} words\n")
            f.write(f"System utterances - Min length: {min(system_lengths)} words\n")
            f.write(f"System utterances - Max length: {max(system_lengths)} words\n")
        
        f.write("\nDOMAIN DISTRIBUTION\n")
        f.write("------------------\n")
        
        domains = stats.get('domains', Counter())
        for domain, count in domains.most_common():
            f.write(f"{domain}: {count} dialogues\n")
        
        f.write("\nSAMPLE CONVERSATIONS\n")
        f.write("-------------------\n")
        
        # Include a few sample conversations
        import random
        sample_indices = random.sample(range(len(conversations)), min(3, len(conversations)))
        
        for i, idx in enumerate(sample_indices):
            conv = conversations[idx]
            f.write(f"\nConversation {i+1} (ID: {conv.get('id', 'unknown')}, Domain: {conv.get('domain', 'unknown')}):\n")
            
            for turn in conv.get('turns', []):
                speaker = turn.get('speaker', '').upper()
                text = turn.get('text', '')
                f.write(f"{speaker}: {text}\n")
            
            f.write("\n")
    
    print(f"Report generated at {report_path}")
    return report_path


def save_processed_data(conversations):
    """
    Save the processed conversations to a JSON file
    """
    output_dir = "./processed_data/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "multiwoz_processed.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(conversations)} processed conversations to {output_path}")
    return output_path


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Explore and analyze the MultiWOZ dataset")
    parser.add_argument("--data_dir", type=str, default="./data/multiwoz", 
                       help="Directory containing the MultiWOZ dataset files")
    
    import sys
    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        # Default arguments when run without command line parameters
        args = parser.parse_args(["--data_dir", "./data/multiwoz"])
    
    # Load dataset files
    files_dict = load_multiwoz_dataset(args.data_dir)
    
    if not files_dict:
        print("Failed to load dataset files. Exiting.")
        return
    
    # Check if data.json exists, if not, look for largest JSON file
    if 'data.json' in files_dict:
        data_json = files_dict['data.json']
    else:
        # Find the largest JSON file which is likely to be the main dialogue file
        largest_file = max(files_dict.items(), key=lambda x: 
                          len(json.dumps(x[1])) if isinstance(x[1], (dict, list)) else 0)
        print(f"data.json not found. Using {largest_file[0]} as the main dialogue file.")
        data_json = largest_file[1]
    
    # Analyze dataset components
    stats = {}
    stats.update(analyze_data_json(data_json))
    analyze_domain_files(files_dict)
    analyze_ontology(files_dict)
    extract_dialogue_acts(files_dict)
    
    # Extract and analyze conversations
    conversations = extract_conversations(data_json)
    stats.update(analyze_conversations(conversations))
    
    # Generate visualizations
    plot_distributions(stats)
    
    # Print sample conversations
    print_sample_conversations(conversations)
    
    # Generate report
    generate_report(stats, conversations)
    
    # Save processed data
    save_processed_data(conversations)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    import argparse
    main()