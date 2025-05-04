# Voice-Enabled Customer Support Chatbot

## Project Overview

This repository contains the early prototype of a voice-enabled customer support assistant that leverages state-of-the-art open-source LLMs (Large Language Models) with speech-to-text and text-to-speech capabilities. The system is designed to provide natural, helpful responses to customer inquiries without requiring proprietary or closed-source models.

### Key Features

- **Fully Open-Source Stack**: Built entirely with open-source components that can be deployed on-premises without external API dependencies
- **LLM Integration**: Uses small-footprint language models capable of running on CPU for testing, with seamless upgrade path to larger models
- **Conversation Memory**: Maintains context across the conversation for more natural interactions
- **Extensible Architecture**: Designed to easily incorporate larger models, knowledge bases, and multi-language support

## Technical Stack

- **Language Models**: TinyLlama, with easy path to upgrade to larger models (Mistral, Zephyr, Llama, etc.)
- **Framework**: LangChain for orchestration and conversation management
- **Speech Processing**: Integration-ready with Whisper (STT) and open TTS systems
- **Python Core**: Built with transformers, PyTorch, and modern Python libraries

## Current Implementation

This repository currently contains a prototype notebook that demonstrates the core conversational capabilities of the system. The notebook:

1. Loads a lightweight language model suitable for CPU environments
2. Creates a synthetic customer support dataset for testing
3. Establishes conversation chains with context memory
4. Simulates customer interactions to test response quality
5. Provides graceful fallback mechanisms when needed

Example conversation:

```
Customer: Hi, can you help me with something?
AI Assistant: Sure! What would you like to ask about?

Customer: How can I create a new account?
AI Assistant: To create an account, please follow these steps:
1. Go to our website and click on the "Sign Up" button at the top right corner of the page. 
2. Fill in your personal details such as name, email address, phone number, etc., and select a password for your account. 
3. Click on the "Create Account" button once you have filled all the required fields. 
That's it! Your account is now created successfully. Thank you for choosing us.
```

## Roadmap

This project is actively under development with the following planned enhancements:

- **FastAPI Backend**: Creating a robust API server to handle voice and text requests
- **Web Interface**: Building a user-friendly frontend for customer interaction
- **Retrieval-Augmented Generation (RAG)**: Incorporating company knowledge bases for more accurate responses
- **Larger Model Integration**: Support for 7B+ parameter models with GPU acceleration
- **Production-Ready Speech Components**: Full integration of Whisper for STT and high-quality TTS systems
- **Multi-Language Support**: Expanding capabilities to handle multiple languages

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Basic understanding of LLMs and conversational AI

## Future Development

I'm actively working on enhancing this system with:

- Switching to FastAPI for production-ready deployment
- Implementing a React frontend with WebRTC for voice capture
- Adding Retrieval-Augmented Generation for better customer-specific responses
- Integrating full speech capabilities for a seamless voice interface
- Fine-tuning language models on domain-specific customer service data

## Contributing

While this is currently a personal portfolio project, I'm open to collaboration. If you're interested in contributing, please reach out to discuss how we might work together.

*This project serves as a demonstration of my skills in machine learning, natural language processing, and full-stack development. It showcases my ability to integrate cutting-edge AI technologies into practical, user-facing applications.*
