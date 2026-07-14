# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Groq that allows users to upload PDF documents and ask questions about their content.

## Features

- Upload PDF documents through intuitive web interface
- Ask questions about PDF content in natural language
- Get accurate, context-aware answers using RAG pipeline
- Persistent vector storage with ChromaDB for data retention
- Fast responses powered by Groq LLM API
- HuggingFace embeddings for semantic search

## Tech Stack

- **Streamlit** - Interactive web application framework
- **LangChain** - LLM orchestration and RAG framework
- **Groq API** - Fast LLM inference
- **ChromaDB** - Vector database for document storage
- **HuggingFace** - Sentence embeddings (all-MiniLM-L6-v2)
- **PyPDF** - PDF document loading and parsing
- **Python-dotenv** - Environment variable management

## Prerequisites

- Python 3.8+
- Groq API key (get it from https://console.groq.com)

## Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd rag-pdf-chatbot
```

2. Create and activate virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create `.env` file and add your Groq API key
```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

1. Run the Streamlit application
```bash
streamlit run app.py
```

2. Open browser to `http://localhost:8501`

3. Upload a PDF document using the file uploader

4. Type your question in the text input

5. Get instant answers from your PDF!

## Project Structure

```
rag-pdf-chatbot/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not in git)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
└── data/
    ├── uploads/                # Uploaded PDF files
    └── vectorstore/            # ChromaDB vector storage
```

## How It Works

1. **Document Loading** - Upload PDF files which are loaded using PyPDFLoader
2. **Text Chunking** - Documents are split into smaller chunks using RecursiveCharacterTextSplitter
3. **Embeddings** - Text chunks are converted to vector embeddings using HuggingFace
4. **Vector Storage** - Embeddings are stored in ChromaDB for fast retrieval
5. **Retrieval** - User query is converted to embedding and similar chunks are retrieved
6. **Generation** - Retrieved chunks + query are sent to Groq LLM for answer generation

## Learning Outcomes

This project demonstrates key Generative AI concepts:
- Retrieval-Augmented Generation (RAG) pattern
- Vector embeddings and semantic search
- Document processing and chunking strategies
- LLM integration with Groq API
- Persistent vector database usage
- Streamlit for rapid AI app prototyping

## Future Enhancements

- Support multiple PDF uploads simultaneously
- Add conversation history and memory
- Implement different chunking strategies
- Add confidence scores for answers
- Support for other document formats (Word, PPT, etc)
- Deploy on cloud platform (Heroku, Streamlit Cloud)

## Troubleshooting

**Issue: Module not found errors**
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`

**Issue: GROQ_API_KEY not found**
- Verify `.env` file exists in root directory
- Check that GROQ_API_KEY is correctly added to `.env`

**Issue: PDF upload fails**
- Ensure PDF file is not corrupted
- Check that file size is reasonable

## License

MIT License

## Author

Built as a learning project for Generative AI fundamentals with LangChain and Streamlit.

