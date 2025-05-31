
# 🧠 Text Embeddings with LangChain

This guide demonstrates how to utilize **OpenAI** and **Hugging Face** embedding models within the **LangChain** framework to convert text into high-dimensional vectors for tasks such as semantic search, clustering, and more.

---

## 📚 Table of Contents

- [OpenAI Embeddings](#openai-embeddings)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Generate Embeddings](#generate-embeddings)
  - [Customize Dimensionality](#customize-dimensionality)
  - [Load and Chunk Documents](#load-and-chunk-documents)
  - [Embed Document Chunks](#embed-document-chunks)
- [Hugging Face Embeddings](#hugging-face-embeddings)
  - [Installation](#installation-1)
  - [Environment Setup](#environment-setup-1)
  - [Generate Embeddings with MiniLM](#generate-embeddings-with-minilm)
  - [Generate Embeddings with MPNet](#generate-embeddings-with-mpnet)
  - [Applications](#applications-1)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

## 🔹 OpenAI Embeddings

### Installation

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

### Environment Setup

Create a `.env` file with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Load your environment variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

---

### Generate Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
text = "This is a tutorial on OPENAI embeddings"
query_result = embeddings.embed_query(text)
len(query_result)
```

---

### Customize Dimensionality

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
```

---

### Load and Chunk Documents

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("speech.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
final_documents = text_splitter.split_documents(docs)
```

---

### Embed Document Chunks

```python
embeddings.embed_query(final_documents[0].page_content)
```

## 🔹 Hugging Face Embeddings
Hugging Face sentence-transformers is a Python framework for state-of-the-art sentence, text, and image embeddings. One of the embedding models is used in the HuggingFaceEmbeddings class. We have also added an alias for SentenceTransformerEmbeddings for users who are more familiar with using the package directly.

### Installation

```bash
pip install langchain langchain-huggingface python-dotenv
```

### Environment Setup

Create a `.env` file with your Hugging Face token:

```env
HF_TOKEN=your_huggingface_access_token
```

Load the token:

```python
from dotenv import load_dotenv
import os

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
```

---

### Generate Embeddings with MiniLM

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_result = embeddings.embed_query("this is a test document")
len(query_result)
```

---

### Generate Embeddings with MPNet

```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectors = embeddings.embed_query("Hello, world!")
len(vectors)
```

---

### Applications

- Semantic Search
- Clustering
- Classification
- Contextual Retrieval

---

## 🙌 Acknowledgements

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Hugging Face Sentence Transformers](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/)
- [Krish Nike](https://krishnaikacademy.com/)
---

## 🧠 Author

**Nahid Zeinali**  
Senior AI Researcher | NLP & LLMs | Healthcare AI  
Follow my journey in MLOps, LLMOps, and Agentic AI.

---
