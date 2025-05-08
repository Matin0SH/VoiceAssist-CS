import json
import os
import random
from sklearn.model_selection import train_test_split

# Create necessary directories
os.makedirs("./processed_data/instruction", exist_ok=True)

# Load the processed data
with open("./processed_data/multiwoz_processed.json", "r", encoding="utf-8") as f:
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

# Split into train, validation, and test
train_data, temp_data = train_test_split(formatted_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Save the splits
with open("./processed_data/instruction/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2)

with open("./processed_data/instruction/validation.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2)

with open("./processed_data/instruction/test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=2)

print(f"Created train set with {len(train_data)} examples")
print(f"Created validation set with {len(val_data)} examples")
print(f"Created test set with {len(test_data)} examples")