
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
# Import Hugging Face Embedding wrapper from LangChain integration
from langchain_huggingface import HuggingFaceEmbeddings

# Import similarity and distance metrics from scikit-learn
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Initialize the embedding model with a lightweight sentence-transformer model
# "all-MiniLM-L6-v2" generates 384-dimensional sentence embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# List of example documents to compare with the query
documents = [
    "what is a capital of USA?",
    "Who is a president of USA?",
    "Who is a prime minister of India?"
]

# Query text to find its similarity with the documents above
query = "Narendra Modi is prime minister of India?"

# Generate embeddings for all documents
doc_vectors = embeddings.embed_documents(documents)

# Generate an embedding for the query
query_vector = embeddings.embed_query(query)

# Compute cosine similarity between the query and each document vector
# Cosine similarity ranges from -1 to 1 (closer to 1 means more similar)
cosine_similarity([query_vector], doc_vectors)

# Compute Euclidean distances between the query and each document vector
# Smaller values indicate higher similarity
euclidean_distances([query_vector], doc_vectors)

```
| Metric            | Similarity Score Range | Behavior                              |
| ----------------- | ---------------------- | ------------------------------------- |
| Cosine Similarity | \[-1, 1]               | Focuses on angle only |
| L2 Distance       | \[0, ∞)                | Focuses on **magnitude + direction**  |
---

## 🗂 FAISS Vector Store

### Building Vector Index

```python
# Import FAISS (Facebook AI Similarity Search) for efficient similarity search
import faiss

# Import FAISS vector store wrapper from LangChain for integration with LangChain framework
from langchain_community.vectorstores import FAISS

# Import an in-memory document store to hold and retrieve documents
from langchain_community.docstore.in_memory import InMemoryDocstore

# Initialize a FAISS index using L2 (Euclidean) distance with 384-dimensional embeddings
# Make sure your embedding model also returns 384-dimensional vectors
index = faiss.IndexFlatL2(384)

# Create a LangChain-compatible FAISS vector store
# - `embedding_function`: a function that converts text into vectors (e.g., OpenAIEmbeddings)
# - `index`: the FAISS index used for similarity search
# - `docstore`: stores documents in memory for retrieval
# - `index_to_docstore_id`: maps FAISS index positions to docstore IDs
vector_store = FAISS(
    embedding_function=embeddings,  # Assume this is defined elsewhere (e.g., OpenAIEmbeddings().embed_query)
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add a list of texts to the vector store
# Each text will be embedded and added to the FAISS index for future retrieval
vector_store.add_texts(["AI is future", "AI is powerful", "Dogs are cute"])

# Perform a similarity search for the query "Tell me about AI"
# `k=3` returns the top 3 most similar documents based on the embedding distance
results = vector_store.similarity_search("Tell me about AI", k=3)

```
| Feature               | `Flat`                | `IVF` (Inverted File Index)        | `HNSW` (Graph-based Index)          |
| --------------------- | --------------------- | ---------------------------------- | ----------------------------------- |
| Type of Search     | Exact                 | Approximate (cluster-based)        | Approximate (graph-based traversal) |
| Speed               | Slow (linear scan)    | Fast (search only in top clusters) | Very Fast (graph walk)              |

---

| Dataset Size | Recommended Index                |
|--------------|----------------------------------|
| ≤ 100K       | `IndexFlatL2`, `IndexFlatIP`     |
| ≤ 1M         | `IndexIVFFlat`, `IndexHNSWFlat`  |
| > 1M         | `IndexIVFPQ`, `IndexHNSWFlat`    |

---

### Metadata-Based Filtering

```python
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

# Step 1: Load the embedding model (MiniLM outputs 384-dim vectors)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 2: Prepare documents with metadata
documents = [
    Document(page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.", metadata={"source": "tweet"}),
    Document(page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.", metadata={"source": "news"}),
    Document(page_content="Building an exciting new project with LangChain - come check it out!", metadata={"source": "tweet"}),
    Document(page_content="Robbers broke into the city bank and stole $1 million in cash.", metadata={"source": "news"}),
    Document(page_content="Wow! That was an amazing movie. I can't wait to see it again.", metadata={"source": "tweet"}),
    Document(page_content="Is the new iPhone worth the price? Read this review to find out.", metadata={"source": "website"}),
    Document(page_content="The top 10 soccer players in the world right now.", metadata={"source": "website"}),
    Document(page_content="LangGraph is the best framework for building stateful, agentic applications!", metadata={"source": "tweet"}),
    Document(page_content="The stock market is down 500 points today due to fears of a recession.", metadata={"source": "news"}),
    Document(page_content="I have a bad feeling I am going to get deleted :(", metadata={"source": "tweet"}),
]

# Step 3: Initialize a FAISS index (inner product similarity - use with normalized vectors)
index = faiss.IndexFlatIP(384)

# Step 4: Initialize LangChain-compatible FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Step 5: Add documents to the FAISS vector store
vector_store.add_documents(documents=documents)

# Step 6: Query to search against the vector store
query = "LangChain provides abstractions to make working with LLMs easy"

# Step 7: Perform unfiltered similarity search (top 2 results)
results_all = vector_store.similarity_search(query, k=2)

# Step 8: Perform metadata-filtered search for "tweets"
results_tweet = vector_store.similarity_search(
    query,
    filter={"source": {"$eq": "tweet"}},
)

# Step 9: Perform metadata-filtered search for "news"
results_news = vector_store.similarity_search(
    query,
    filter={"source": {"$eq": "news"}},
)

# Step 10: Print example result from news
print("Top result from 'news' source:")
print("Content:", results_news[0].page_content)
print("Metadata:", results_news[0].metadata)

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
