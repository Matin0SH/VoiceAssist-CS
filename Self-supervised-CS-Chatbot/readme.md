# Self-Supervised Customer Service Chatbot

This repository contains implementation code for building a self-supervised customer service chatbot based on the blueprint provided. The chatbot leverages Large Language Models with parameter-efficient fine-tuning and can be continuously improved using techniques like Reinforcement Learning from Human Feedback (RLHF).

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Step-by-Step Guide](#step-by-step-guide)
   - [Step 1: Dataset Collection](#step-1-dataset-collection)
   - [Step 2: Dataset Exploration](#step-2-dataset-exploration)
   - [Step 3: Data Preprocessing](#step-3-data-preprocessing)
   - [Step 4: Model Training](#step-4-model-training)
   - [Step 5: Testing the Chatbot](#step-5-testing-the-chatbot)
5. [Advanced Implementation](#advanced-implementation)
6. [Troubleshooting](#troubleshooting)

## Overview

This implementation follows the blueprint for a self-supervised customer service chatbot. The key components include:

- **Dataset Loader**: Scripts to download and load publicly available customer service conversation datasets.
- **Dataset Explorer**: Tools to analyze and visualize dataset characteristics.
- **Data Preprocessor**: Converts raw conversations into formats suitable for model training.
- **Model Trainer**: Fine-tunes large language models using parameter-efficient methods (LoRA).
- **Inference Server**: A simple web interface to interact with the trained chatbot.

The implementation uses a modular approach, allowing you to replace or enhance individual components as needed.

## Requirements

### Hardware Requirements
- GPU with at least 8GB VRAM for training (16GB+ recommended)
- 16GB+ RAM
- 100GB+ disk space for datasets and models

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (Parameter-Efficient Fine-Tuning)
- Flask (for the inference server)
- NVIDIA CUDA 11.7+ (for GPU acceleration)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/customer-service-chatbot.git
cd customer-service-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

The requirements.txt file should contain:
```
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
datasets>=2.12.0
tqdm>=4.65.0
nltk>=3.8.1
scikit-learn>=1.2.2
pandas>=2.0.0
matplotlib>=3.7.1
seaborn>=0.12.2
flask>=2.3.2
bitsandbytes>=0.38.0
accelerate>=0.20.0
tensorboard>=2.13.0
wandb>=0.15.4
```

## Step-by-Step Guide

### Step 1: Dataset Collection

We'll start by collecting customer service conversation datasets that don't require API keys.

```bash
python dataset_loader.py
```

This script will:
1. Download several public customer service datasets (MultiWOZ, MSDialog, Taskmaster, Schema-Guided Dialogue)
2. Process and standardize the format
3. Save the processed datasets to a directory called `processed_data`

You'll be prompted to select which datasets to download. For a first run, we recommend selecting all available options.

### Step 2: Dataset Exploration

Next, let's explore the datasets to better understand their characteristics:

```bash
python dataset_explorer.py
```

This script will:
1. Generate statistics about each dataset
2. Create visualizations for conversation length, utterance length, and common intents
3. Save detailed reports in a directory called `reports`
4. Optionally combine datasets into a single unified dataset

Review the generated reports and visualizations to understand your training data better.

### Step 3: Data Preprocessing

Now, let's preprocess the data for model training:

```bash
python dataset_preprocessor.py --input ./processed_data/combined_dataset.json --output_dir ./training_data --format instruction,chat,context --multi_turn
```

This script will:
1. Clean and normalize text in conversations
2. Extract instruction-response pairs
3. Format data for different training approaches (instruction, chat, context)
4. Create train/validation/test splits
5. Save processed data to the specified output directory

You can adjust the following parameters:
- `--input`: Path to the input dataset JSON file
- `--output_dir`: Output directory for processed data
- `--format`: Comma-separated list of format types (instruction, chat, context)
- `--multi_turn`: Enable multi-turn context extraction
- `--max_history`: Maximum number of turns to include in context (default: 3)

### Step 4: Model Training

Now, let's train the customer service chatbot model:

```bash
python chatbot_trainer.py --model_name mistralai/Mistral-7B-v0.1 --data_dir ./training_data --output_dir ./fine_tuned_model --format_type instruction --batch_size 4 --num_epochs 3 --use_4bit
```

This script will:
1. Load the base language model
2. Set up parameter-efficient fine-tuning with LoRA
3. Train the model on the processed data
4. Evaluate the model on test data
5. Save the fine-tuned model to the specified output directory

You can adjust the following parameters:
- `--model_name`: Base model to fine-tune (e.g., meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1)
- `--data_dir`: Directory containing processed training data
- `--output_dir`: Output directory for the fine-tuned model
- `--format_type`: Format type of the training data (instruction, chat, context)
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--use_4bit`: Use 4-bit quantization (recommended for larger models)
- `--use_8bit`: Use 8-bit quantization

**Note**: Training can take several hours depending on your GPU. For faster results, you can reduce the number of epochs or use a smaller model.

### Step 5: Testing the Chatbot

Finally, let's test the chatbot with a simple web interface:

```bash
python chatbot_server.py --model_path ./fine_tuned_model
```

This script will:
1. Load the fine-tuned model
2. Start a web server with a chat interface
3. Allow you to interact with the chatbot and adjust generation parameters

You can access the web interface by opening http://localhost:5000 in your browser.

You can adjust the following parameters:
- `--model_path`: Path to the fine-tuned model
- `--port`: Port to run the server on (default: 5000)
- `--host`: Host to run the server on (default: 0.0.0.0)
- `--device`: Device to run the model on (cuda or cpu)

## Advanced Implementation

To implement the full self-supervised learning pipeline as described in the blueprint, consider the following enhancements:

1. **RLHF Integration**:
   - Implement a preference data collection system
   - Train a reward model on human preferences
   - Use PPO or DPO to optimize the model based on the reward function

2. **Self-Improvement Loop**:
   - Create an automated query generation system
   - Implement a teacher model for evaluating responses
   - Set up continuous training with feedback incorporation

3. **Knowledge Integration**:
   - Build a document ingestion pipeline
   - Implement vector-based knowledge retrieval
   - Create a knowledge graph for relationship modeling

4. **Domain Adaptation**:
   - Develop a few-shot adaptation module
   - Create personality configuration systems
   - Implement domain-specific evaluation metrics

These advanced features require additional components that build upon the base implementation provided here.

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size
   - Use 4-bit or 8-bit quantization (--use_4bit or --use_8bit flags)
   - Use a smaller model
   - Reduce sequence length (--max_length parameter)

2. **Slow Training**:
   - Ensure you're using GPU acceleration
   - Use gradient accumulation to simulate larger batch sizes
   - Reduce dataset size for initial experiments

3. **Poor Generation Quality**:
   - Increase training epochs
   - Check data quality with the dataset explorer
   - Try different format types (instruction, chat, context)
   - Adjust generation parameters (temperature, top_p)

4. **CUDA Issues**:
   - Ensure CUDA toolkit version is compatible with PyTorch
   - Try setting CUDA_VISIBLE_DEVICES if you have multiple GPUs

### Getting Help

If you encounter issues not covered here, please:
1. Check existing issues in the repository
2. Provide detailed information about your environment and the specific error
3. Include relevant logs and error messages
