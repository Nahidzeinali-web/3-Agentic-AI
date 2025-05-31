
# 🧠 Text Embeddings and Retrieval with LangChain, OpenAI, Hugging Face, FAISS & Traditional RAG

This comprehensive guide walks you through:
- Generating embeddings with OpenAI and Hugging Face using LangChain
- Comparing similarity using distance metrics
- Implementing vector search with FAISS
- Performing metadata-filtered retrieval
- Building RAG pipelines using Gemini and LangChain

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
  - [Embedding Queries and Similarity](#embedding-queries-and-similarity)
- [FAISS Vector Store](#faiss-vector-store)
  - [Building Vector Index](#building-vector-index)
  - [Metadata-Based Filtering](#metadata-based-filtering)
  - [Saving and Reloading Index](#saving-and-reloading-index)
- [PDF Loading and Chunking](#pdf-loading-and-chunking)
- [RAG Pipeline with Gemini and LangChain](#rag-pipeline-with-gemini-and-langchain)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

## 🔹 OpenAI Embeddings

### Installation

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

### Environment Setup

```env
OPENAI_API_KEY=your_openai_api_key_here
```

```python
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

### Generate Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
query_result = embeddings.embed_query("This is a tutorial on OPENAI embeddings")
len(query_result)
```

### Customize Dimensionality

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
```

### Load and Chunk Documents

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("speech.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
final_documents = splitter.split_documents(docs)
```

### Embed Document Chunks

```python
embeddings.embed_query(final_documents[0].page_content)
```

---

## 🔹 Hugging Face Embeddings

### Installation

```bash
pip install langchain langchain-huggingface python-dotenv
```

### Environment Setup

```env
HF_TOKEN=your_huggingface_access_token
```

```python
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
```

### Embedding Queries and Similarity

```python
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents = ["what is a capital of USA?",
             "Who is a president of USA?",
             "Who is a prime minister of India?"]

query = "Narendra modi is prime minister of india?"
doc_vectors = embeddings.embed_documents(documents)
query_vector = embeddings.embed_query(query)

cosine_similarity([query_vector], doc_vectors)
euclidean_distances([query_vector], doc_vectors)
```

---

## 🗂 FAISS Vector Store

### Building Vector Index

```python
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

index = faiss.IndexFlatL2(384)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
vector_store.add_texts(["AI is future", "AI is powerful", "Dogs are cute"])
results = vector_store.similarity_search("Tell me about AI", k=3)
```

| Dataset Size | Recommended Index                |
|--------------|----------------------------------|
| ≤ 100K       | `IndexFlatL2`, `IndexFlatIP`     |
| ≤ 1M         | `IndexIVFFlat`, `IndexHNSWFlat`  |
| > 1M         | `IndexIVFPQ`, `IndexHNSWFlat`    |

---

### Metadata-Based Filtering

```python
from langchain_core.documents import Document

docs = [Document(page_content="...", metadata={"source": "tweet"}), ...]
vector_store.add_documents(docs)

vector_store.similarity_search("LangChain...", k=2)
vector_store.similarity_search("LangChain...", filter={"source": {"$eq": "tweet"}})
```

---

### Saving and Reloading Index

```python
vector_store.save_local("todays_class_faiss_index")
new_store = FAISS.load_local("todays_class_faiss_index", embeddings, allow_dangerous_deserialization=True)
new_store.similarity_search("langchain")
```

---

## 📄 PDF Loading and Chunking

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("llama2.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(pages)

index = faiss.IndexFlatIP(384)
pdf_vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
pdf_vector_store.add_documents(split_docs)
```

---

## 🤖 RAG Pipeline with Gemini and LangChain

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = pdf_vector_store.as_retriever(search_kwargs={"k": 10})
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

rag_chain.invoke("what is llama model?")
```

---

## 🙌 Acknowledgements

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Hugging Face Sentence Transformers](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/)
- [Krish Naik Academy](https://krishnaikacademy.com/)

---

## 👨‍💻 Author

**Nahid Zeinali**  
Senior AI Researcher | NLP & LLMs | Healthcare AI  
Follow my journey in MLOps, LLMOps, and Agentic AI.

---
