
# 🧠 Text Embedding with LangChain and OpenAI

This project demonstrates how to convert raw text into high-dimensional vector embeddings using [OpenAI's `text-embedding-3-large`](https://platform.openai.com/docs/guides/embeddings) model integrated with the [LangChain framework](https://python.langchain.com/). These embeddings can be used in downstream tasks such as semantic search, clustering, similarity measurement, and as input to other NLP models.

---

## 📦 Requirements

Install dependencies:

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

Also, create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 📘 Overview

The notebook (or Python script) covers:

- Setting up OpenAI Embeddings
- Embedding individual queries
- Controlling vector dimensionality
- Loading and splitting documents
- Generating embeddings from document chunks

---

## 🚀 Getting Started

### 1. ✅ Load Environment Variables

```python
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

We use `python-dotenv` to securely load the OpenAI API key from an `.env` file.

---

### 2. 📐 Generate Embeddings for Text

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

This creates an embedding instance using OpenAI’s `text-embedding-3-large` model, which converts text into a high-dimensional vector.

To embed a sample text:

```python
text = "This is a tutorial on OPENAI embeddings"
query_result = embeddings.embed_query(text)
```

The result is a vector of floating-point numbers (usually 1536 or a custom dimension).

To see the embedding length:

```python
len(query_result)
```

---

### 3. 🛠 Customize Embedding Dimensionality

OpenAI allows specifying the output dimensions (e.g., 1024 instead of default 1536):

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
```

This can be helpful when dimensionality reduction is desired for storage or performance.

---

### 4. 📄 Load a Text File as a Document

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("speech.txt")
docs = loader.load()
```

This loads the content of a `.txt` file into a LangChain `Document` object.

---

### 5. ✂️ Split Text into Chunks

To make large texts manageable, we split them into smaller chunks using the `RecursiveCharacterTextSplitter`:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
final_documents = text_splitter.split_documents(docs)
```

- `chunk_size`: max characters per chunk
- `chunk_overlap`: number of overlapping characters to preserve context

View the first chunk:

```python
final_documents[0].page_content
```

---

### 6. 🔎 Embed Document Chunks

Finally, we embed the first document chunk:

```python
embeddings.embed_query(final_documents[0].page_content)
```

Each chunk is turned into a vector, which can be stored or used for semantic retrieval, clustering, or classification.

---

## 📊 Applications of Text Embeddings

- **Semantic Search**: Match questions to relevant answers or documents.
- **Clustering**: Group similar texts using k-means or HDBSCAN.
- **Classification**: Use vector inputs for downstream ML models.
- **Recommendation Systems**: Match user queries to best-suited documents.

---

## 📁 File Structure

```
├── speech.txt                  # Sample input text
├── .env                        # Your OpenAI API key
├── embedding_script.py         # This code as a Python file
└── README.md                   # You are here
```

---

## 🙌 Acknowledgements

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [LangChain Python Docs](https://python.langchain.com/)

---

## 🧠 Author

**Nahid Zeinali**  
Senior AI Researcher | NLP & LLMs | Healthcare AI  
Follow my learning journey through MLOps, LLMOps, and Agentic AI.

---
