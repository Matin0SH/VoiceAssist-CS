"""
Customer Service Chatbot Server
A simple inference server for the customer service chatbot
"""

import os
import json
import torch
import argparse
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import time


# HTML template for the chat interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Customer Service Chatbot</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .chat-container {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            height: 400px;
            overflow-y: auto;
            background-color: #f9f9f9;
        }
        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 6px;
            max-width: 70%;
            word-wrap: break-word;
        }
        .user {
            background-color: #e3f2fd;
            margin-left: auto;
            text-align: right;
        }
        .assistant {
            background-color: #f0f0f0;
            margin-right: auto;
        }
        .input-area {
            display: flex;
            margin-top: 20px;
        }
        #user-input {
            flex: 1;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            margin-left: 10px;
            padding: 8px 16px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        #thinking {
            display: none;
            color: #888;
            font-style: italic;
            margin-top: 10px;
        }
        .system-controls {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 4px;
        }
        .settings-group {
            margin-bottom: 10px;
        }
        label {
            margin-right: 10px;
        }
        .title {
            text-align: center;
            color: #333;
        }
        .clear-btn {
            background-color: #f44336;
        }
        .clear-btn:hover {
            background-color: #d32f2f;
        }
    </style>
</head>
<body>
    <h1 class="title">Customer Service Chatbot</h1>
    
    <div class="system-controls">
        <div class="settings-group">
            <label for="temperature">Temperature:</label>
            <input type="range" id="temperature" min="0.1" max="1.0" step="0.1" value="0.7">
            <span id="temp-value">0.7</span>
        </div>
        
        <div class="settings-group">
            <label for="max-length">Max Length:</label>
            <input type="number" id="max-length" min="50" max="1000" value="200">
        </div>
        
        <div class="settings-group">
            <label for="top-p">Top P:</label>
            <input type="range" id="top-p" min="0.1" max="1.0" step="0.1" value="0.9">
            <span id="top-p-value">0.9</span>
        </div>
    </div>
    
    <div class="chat-container" id="chat-container">
        <div class="message assistant">Hello! I'm a customer service assistant. How can I help you today?</div>
    </div>
    
    <div id="thinking">Thinking...</div>
    
    <div class="input-area">
        <input type="text" id="user-input" placeholder="Type your message here..." autocomplete="off">
        <button id="send-btn">Send</button>
        <button id="clear-btn" class="clear-btn">Clear</button>
    </div>
    
    <script>
        // Elements
        const chatContainer = document.getElementById('chat-container');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const clearBtn = document.getElementById('clear-btn');
        const thinking = document.getElementById('thinking');
        const temperatureSlider = document.getElementById('temperature');
        const temperatureValue = document.getElementById('temp-value');
        const maxLengthInput = document.getElementById('max-length');
        const topPSlider = document.getElementById('top-p');
        const topPValue = document.getElementById('top-p-value');
        
        // Update slider values
        temperatureSlider.addEventListener('input', () => {
            temperatureValue.textContent = temperatureSlider.value;
        });
        
        topPSlider.addEventListener('input', () => {
            topPValue.textContent = topPSlider.value;
        });
        
        // Chat history
        let chatHistory = [];
        
        // Send message function
        async function sendMessage() {
            const userMessage = userInput.value.trim();
            if (!userMessage) return;
            
            // Add user message to chat
            addMessage('user', userMessage);
            userInput.value = '';
            
            // Update chat history
            chatHistory.push({role: 'user', content: userMessage});
            
            // Show thinking indicator
            thinking.style.display = 'block';
            
            try {
                // Get generation parameters
                const temperature = parseFloat(temperatureSlider.value);
                const maxLength = parseInt(maxLengthInput.value);
                const topP = parseFloat(topPSlider.value);
                
                // Send request to server
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        messages: chatHistory,
                        temperature: temperature,
                        max_length: maxLength,
                        top_p: topP
                    })
                });
                
                const data = await response.json();
                
                // Add assistant response to chat
                addMessage('assistant', data.response);
                
                // Update chat history
                chatHistory.push({role: 'assistant', content: data.response});
            } catch (error) {
                console.error('Error:', error);
                addMessage('assistant', 'Sorry, there was an error processing your request.');
            } finally {
                // Hide thinking indicator
                thinking.style.display = 'none';
            }
        }
        
        // Add message to chat
        function addMessage(role, content) {
            const messageElement = document.createElement('div');
            messageElement.classList.add('message', role);
            messageElement.textContent = content;
            
            chatContainer.appendChild(messageElement);
            
            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Event listeners
        sendBtn.addEventListener('click', sendMessage);
        
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        clearBtn.addEventListener('click', () => {
            // Clear chat history
            chatHistory = [];
            
            // Clear chat container but keep the welcome message
            chatContainer.innerHTML = '<div class="message assistant">Hello! I\'m a customer service assistant. How can I help you today?</div>';
        });
    </script>
</body>
</html>
"""


class ChatbotServer:
    def __init__(self, model_path, device="cuda"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"Loading model on {self.device}...")
        
        # Load configuration
        self.config = PeftConfig.from_pretrained(model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print(f"Loading base model: {self.config.base_model_name_or_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name_or_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Load LoRA weights
        print("Loading LoRA weights...")
        self.model = PeftModel.from_pretrained(self.model, model_path)
        
        if self.device == "cpu":
            self.model.to(self.device)
        
        print("Model loaded successfully!")
    
    def generate_response(self, messages, temperature=0.7, max_length=200, top_p=0.9):
        """
        Generate a response for a given list of messages
        """
        # Format messages into prompt
        prompt = "<s>"
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                prompt += f"[INST] {content} [/INST] "
            elif role == "assistant" and i < len(messages) - 1:  # Don't include the last assistant message in the prompt
                prompt += f"{content}</s>"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the response (remove the input prompt)
        response = generated_text[len(prompt):]
        
        # Clean up any remaining special tokens or formatting
        response = response.replace("</s>", "").strip()
        
        print(f"Generation time: {generation_time:.2f} seconds")
        
        return response


# Create Flask app
app = Flask(__name__)
chatbot = None


@app.route('/')
def home():
    """
    Render the chat interface
    """
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    """
    Generate a response for a given message
    """
    data = request.json
    messages = data.get('messages', [])
    temperature = data.get('temperature', 0.7)
    max_length = data.get('max_length', 200)
    top_p = data.get('top_p', 0.9)
    
    # Validate inputs
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    
    # Format messages
    formatted_messages = []
    for message in messages:
        formatted_messages.append({
            "role": message.get("role"),
            "content": message.get("content")
        })
    
    try:
        response = chatbot.generate_response(
            formatted_messages,
            temperature=temperature,
            max_length=max_length,
            top_p=top_p
        )
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error generating response: {e}")
        return jsonify({"error": str(e)}), 500


def main():
    """
    Main function to start the server
    """
    parser = argparse.ArgumentParser(description="Start a customer service chatbot server")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run the model on")
    
    args = parser.parse_args()
    
    global chatbot
    chatbot = ChatbotServer(args.model_path, device=args.device)
    
    print(f"Starting server on {args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()