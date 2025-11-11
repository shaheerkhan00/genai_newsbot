#  News Research Tool 📈

News Bot is an AI-powered news research tool that processes news articles from URLs and allows you to ask questions about their content using Retrieval-Augmented Generation (RAG).

## Features

- **URL Processing**: Load and process multiple news article URLs simultaneously
- **AI-Powered Q&A**: Ask questions about the content of processed articles
- **Source Tracking**: See which sources provided the information for each answer
- **Fast Processing**: Uses Groq's high-speed LLM inference
- **Local Vector Store**: Saves processed embeddings for future use

## Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - LangChain (RAG pipeline)
  - Groq (LLM inference)
  - Sentence Transformers (embeddings)
  - FAISS (vector storage)
- **Document Processing**: Unstructured

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/finance_articles_genai.git
   cd finance_articles_genai
