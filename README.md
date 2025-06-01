# 🧠 Text Embeddings and Retrieval with LangChain, OpenAI, Hugging Face, FAISS, Pinecone, Gemini, LangChain Hub Prompt  & Traditional RAG

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
  - [Beginner’s Guide to PDF Q&A with RAG using LangChain, FAISS, and GPT-4](#beginners-guide-to-pdf-qa-with-rag-using-langchain-faiss-and-gpt-4)
  - [Full RAG Learning Script with HuggingFace, Pinecone, Gemini, and LangChain Hub Prompt](#full-rag-learning-script-with-huggingface-pinecone-gemini-and-langchain-hub-prompt)

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

# Import the FAISS vector store wrapper from LangChain for integration with the LangChain framework
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
    embedding_function=embeddings,  # Assume this is defined elsewhere (e.g., OpenAIEmbeddings() embed_query)
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add a list of texts to the vector store
# Each text will be embedded and added to the FAISS index for future retrieval
vector_store.add_texts(["AI is the future", "AI is powerful", "Dogs are cute"])

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

## 🧠 Tips

- Use `IndexFlatIP` if your vectors are normalized (for cosine similarity).
- Use `IndexHNSWFlat` for good accuracy without training, even at a large scale.
- Use `IndexIVFPQ` when working with **very large datasets** and need **compressed storage**.
---

### Full Example Code--> Metadata-Based Filtering

```python
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import faiss

# Step 1: Load Hugging Face embedding model (MiniLM: 384-dim output)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 2: Prepare sample documents with metadata
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

# Step 3: Create a FAISS index using inner product (for cosine similarity, use normalized vectors)
index = faiss.IndexFlatIP(384)

# Step 4: Wrap FAISS index with LangChain's vector store interface
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Step 5: Add documents to the vector store
vector_store.add_documents(documents=documents)

# Step 6: Define query
query = "LangChain provides abstractions to make working with LLMs easy"

# Step 7: Unfiltered similarity search (top 2)
results_all = vector_store.similarity_search(query, k=2)

# Step 8: Metadata-filtered search: tweets only
results_tweet = vector_store.similarity_search(
    query,
    filter={"source": {"$eq": "tweet"}},
)

# Step 9: Metadata-filtered search: news only
results_news = vector_store.similarity_search(
    query,
    filter={"source": {"$eq": "news"}},
)

# Step 10: Print result from news filter
print("Top result from 'news' source:")
print("Content:", results_news[0].page_content)
print("Metadata:", results_news[0].metadata)

# Step 11: Create retriever interface (k=3)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Step 12: Invoke the retriever
retrieved_docs = retriever.invoke("LangChain provides abstractions to make working with LLMs easy")
```
# 🧠 Vector Store Storage Strategies: In-Memory, On-Disk, and Cloud

When building semantic search or Retrieval-Augmented Generation (RAG) systems using LangChain and FAISS (or other vector databases), it's essential to select the appropriate **storage backend** based on your specific requirements for speed, persistence, and scalability.


1. In-Memory Vector Store: Stores all document vectors and metadata directly in system memory (RAM) using LangChain’s `InMemoryDocstore`.
2. On-Disk Vector Store: Stores the FAISS index and metadata on disk using faiss.write_index() and faiss.read_index(). Enables persistence between runs.
3. Cloud Vector Store: Utilizes managed vector databases, such as Pinecone, Qdrant, Weaviate, Vespa, or Chroma, to store and serve embeddings remotely via API.

---
## 🧩 Overview

| Type        | Description                           | Speed      | Persistence | Memory Usage | Scalability |
|-------------|---------------------------------------|------------|-------------|---------------|-------------|
| In-Memory   | Stores everything in RAM              | ⚡ Very Fast | ❌ No       | 🔺 High        | ⚠️ Limited   |
| On-Disk     | Saves to local file system            | ✅ Moderate | ✅ Yes      | ✅ Low         | ✅ Good      |
| Cloud       | Remote storage via API (e.g., Pinecone, Qdrant) | 🌐 Varies  | ✅ Yes      | ⚡ Flexible     | ✅ Excellent |

---


### Saving and Reloading Index

```python
vector_store.save_local("todays_class_faiss_index")
new_store = FAISS.load_local("todays_class_faiss_index", embeddings, allow_dangerous_deserialization=True)
new_store.similarity_search("langchain")
```

---

# 🧠 Beginner’s Guide to PDF Q&A with RAG using LangChain, FAISS, and GPT-4

This script lets you **ask questions about a PDF file**, and get grounded answers using **GPT-4** and **retrieved content from the PDF**. Here's a detailed explanation of each step:

---

### ✅ Step 1: Import Required Libraries  
We load all necessary tools:
- `LangChain` modules to work with documents, split text, and run models.
- `FAISS` for fast document similarity search.
- `Hugging Face` for embeddings.
- `OpenAI` for language model (GPT-4o).

---

### ✅ Step 2: Define PDF File Path  
```python
FILE_PATH = "llama2.pdf"
```
Specify the PDF you want to read and ask questions about.

---

### ✅ Step 3: Load PDF Pages  
```python
loader = PyPDFLoader(FILE_PATH)
pages = loader.load()
```
This reads the PDF and loads each page into memory so we can process them.

---

### ✅ Step 4: Split Pages into Chunks  
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(pages)
```
AI models work better with short, focused input. So we split each page into:
- Chunks of 500 characters.
- With 50-character overlaps to preserve flow between chunks.

---

### ✅ Step 5: Convert Text Chunks into Vectors (Embeddings)  
```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```
Each chunk is turned into a **vector** (list of numbers) using a pre-trained model (`MiniLM`).
This lets us compare chunks to questions later.

---

### ✅ Step 6: Create a FAISS Index for Similarity Search  
```python
index = faiss.IndexFlatIP(384)
```
We create a FAISS index, which is like a searchable database of vectors.
`IP` (inner product) is used for cosine similarity.

---

### ✅ Step 7: Wrap FAISS in LangChain Vector Store  
```python
pdf_vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
```
This combines our embeddings and index so we can:
- Add documents.
- Retrieve them by similarity.

---

### ✅ Step 8: Add Document Vectors to the Store  
```python
pdf_vector_store.add_documents(split_docs)
```
Now all your chunks are searchable through vector similarity.

---

### ✅ Step 9: Create a Retriever  
```python
retriever = pdf_vector_store.as_retriever(search_kwargs={"k": 10})
```
This tool finds the **top 10 most relevant chunks** based on a user's question.

---

### ✅ Step 10: Load a Prompt Template from LangChain Hub  
```python
prompt = hub.pull("rlm/rag-prompt")
```
This is a predefined template that tells GPT-4 how to answer questions using retrieved context.

---

### ✅ Step 11: Define Helper to Format Retrieved Text  
```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
```
This formats the retrieved chunks into a single string so GPT-4 can read them.

---

### ✅ Step 12: Build the RAG Chain  
```python
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)
```
This connects:
- Your question ➡️
- To retrieved context ➡️
- Into GPT-4 with a template ➡️
- And returns a nice answer.

---

### ✅ Step 13: Ask a Question and Get an Answer  
```python
response = rag_chain.invoke("What is the LLaMA model?")
print(response)
```
This runs the pipeline and prints GPT-4's answer based on what it found in the PDF.

---
✨ You're now using **Retrieval-Augmented Generation (RAG)** to query PDFs with competent, grounded answers!
---
## 🤖 FULL CODE
```python
# 1. Import required libraries

# Load PDF files using PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

# Split long text into smaller overlapping chunks for better embedding and retrieval
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS is a library for efficient similarity search of embeddings
from langchain_community.vectorstores import FAISS

# In-memory storage for document chunks used by LangChain vector store
from langchain_community.docstore.in_memory import InMemoryDocstore

# Use MiniLM model for computing dense vector embeddings for chunks
from langchain_huggingface import HuggingFaceEmbeddings

# ChatOpenAI allows using OpenAI’s GPT model through LangChain
from langchain_openai import ChatOpenAI

# LangChain Hub for pulling pre-defined prompts (e.g., RAG prompt template)
from langchain import hub

# Utility to parse the model’s output into a string
from langchain_core.output_parsers import StrOutputParser

# Used for direct passthrough of input values (e.g., the user question)
from langchain_core.runnables import RunnablePassthrough

# Direct use of FAISS backend for indexing and similarity search
import faiss

# Pretty print utility (optional, useful for inspecting objects)
import pprint


# 2. Define the path to the input PDF document
FILE_PATH = "llama2.pdf"


# 3. Load and split the PDF into chunks

# Load the content of the PDF as LangChain Documents (each page as a Document)
loader = PyPDFLoader(FILE_PATH)
pages = loader.load()

# Split each page into smaller chunks of 500 characters with 50 character overlap
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(pages)


# 4. Initialize the embedding model (MiniLM-L6-v2, 384-dimensional)

# This HuggingFace embedding model converts text into dense vectors for semantic search
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# 5. Set up a FAISS index

# FAISS index configured for inner product (dot product), commonly used with normalized vectors
dimension = 384  # MiniLM outputs 384-dimensional vectors
index = faiss.IndexFlatIP(dimension)


# 6. Create a LangChain-compatible FAISS vector store

# InMemoryDocstore keeps track of which vector maps to which document
# index_to_docstore_id is initially empty and will be populated automatically
pdf_vector_store = FAISS(
    embedding_function=embeddings,       # Function to convert text into embeddings
    index=index,                          # The FAISS index to use
    docstore=InMemoryDocstore(),         # Stores the actual document chunks
    index_to_docstore_id={}              # Maps FAISS index entries to document IDs
)


# 7. Add the split document chunks to the FAISS vector store

# This indexes all chunked documents using the embeddings and stores them in FAISS
pdf_vector_store.add_documents(split_docs)


# 8. Create a retriever to get the top-10 most relevant chunks for a query

# The retriever is used to fetch the top-k semantically similar chunks given a question
retriever = pdf_vector_store.as_retriever(search_kwargs={"k": 10})


# 9. Load the RAG (Retrieval-Augmented Generation) prompt template from LangChain Hub

# This is a pre-configured prompt designed for RAG-style interactions
prompt = hub.pull("rlm/rag-prompt")

# (Optional) Inspect the full prompt structure
# pprint.pprint(prompt.messages)


# 10. Helper function to format the retrieved documents

# Joins multiple document chunks into a single string separated by double line breaks
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 11. Define the complete RAG pipeline

# Combines:
# - Retriever to fetch relevant chunks
# - Prompt to wrap question and context
# - OpenAI’s GPT-4o model to generate the answer
# - OutputParser to extract plain string output

rag_chain = (
    {
        "context": retriever | format_docs,       # Chain: retrieve → format as text
        "question": RunnablePassthrough()         # Directly passes the input question
    }
    | prompt                                      # Applies the RAG prompt
    | ChatOpenAI(model="gpt-4o")                  # GPT-4o model for response generation
    | StrOutputParser()                           # Extracts the generated answer string
)


# 12. Provide a query to the pipeline

# Example question to ask the RAG system
query = "What is the LLaMA model?"
response = rag_chain.invoke(query)


# 13. Print the model’s response

print("\n🧠 Answer from RAG Pipeline:")
print(response)

```
---
## Full RAG Learning Script with HuggingFace, Pinecone, Gemini, and LangChain Hub Prompt
```python
# ------------------------------------------------------------------------------------
# 📄 Step 1: Create LangChain Document Objects with Metadata and UUIDs
# ------------------------------------------------------------------------------------

from uuid import uuid4  # Used to create unique IDs for each document
from langchain_core.documents import Document  # LangChain document object

# Create a list of (text, source) tuples as sample documents
raw_docs = [
    ("I had chocolate chip pancakes and scrambled eggs for breakfast this morning.", "tweet"),
    ("The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.", "news"),
    ("Building an exciting new project with LangChain - come check it out!", "tweet"),
    ("Robbers broke into the city bank and stole $1 million in cash.", "news"),
    ("Wow! That was an amazing movie. I can't wait to see it again.", "tweet"),
    ("Is the new iPhone worth the price? Read this review to find out.", "website"),
    ("The top 10 soccer players in the world right now.", "website"),
    ("LangGraph is the best framework for building stateful, agentic applications!", "tweet"),
    ("The stock market is down 500 points today due to fears of a recession.", "news"),
    ("I have a bad feeling I am going to get deleted :(", "tweet"),
]

documents = []  # Will store LangChain Document objects
uuids = []  # Will store UUIDs for indexing in Pinecone

# Convert each raw document into a LangChain Document with a UUID and metadata
for text, src in raw_docs:
    uid = str(uuid4())  # Generate a unique ID
    uuids.append(uid)  # Save UUID to associate later
    documents.append(Document(page_content=text, metadata={"source": src, "id": uid}))  # Create document

# Print each document’s UUID and source type for debugging and traceability
for i, doc in enumerate(documents, 1):
    print(f"{i:02d}. UUID: {doc.metadata['id']} — Source: {doc.metadata['source']}")

# ------------------------------------------------------------------------------------
# 🌐 Step 2: Pinecone Setup and HuggingFace Embeddings (384-D)
# ------------------------------------------------------------------------------------

import os  # OS operations
from dotenv import load_dotenv  # Load API keys from .env
from pinecone import Pinecone, ServerlessSpec  # Pinecone client setup
from langchain_pinecone import PineconeVectorStore  # LangChain Pinecone wrapper
from langchain_huggingface import HuggingFaceEmbeddings  # HuggingFace embedding loader

load_dotenv()  # Load variables from .env file
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Get Pinecone API key

pc = Pinecone(api_key=pinecone_api_key)  # Connect to Pinecone
index_name = "firstproject"  # Name of the index
embedding_dim = 384  # all-MiniLM-L6-v2 has a fixed 384-dim output

# Delete old index (if exists) to prevent dimension mismatch issues
if pc.has_index(index_name):
    print(f"⚠️ Deleting old index '{index_name}' for fresh start...")
    pc.delete_index(index_name)

# Create a new index with correct embedding dimension and cosine similarity
print("✅ Creating Pinecone index...")
pc.create_index(
    name=index_name,
    dimension=embedding_dim,
    metric="cosine",  # Similarity metric
    spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Serverless region setup
)

# Load embedding model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Check the actual output dimension to validate correctness
print("✅ Confirmed embedding dimension:", len(embeddings.embed_query("test")))

# Connect to the Pinecone index and initialize LangChain vector store
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Add the documents into Pinecone with associated UUIDs
print("📥 Adding documents to vector store...")
vector_store.add_documents(documents=documents, ids=uuids)

# ------------------------------------------------------------------------------------
# 🔍 Step 3: Setup Retriever with Score Threshold
# ------------------------------------------------------------------------------------

# Define retriever that filters out low-relevance matches using a score threshold
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",  # Match if above threshold
    search_kwargs={"score_threshold": 0.7}  # Minimum score to consider relevant
)

# ------------------------------------------------------------------------------------
# 🧠 Step 4: Setup LangChain RAG with Google Gemini & Prompt Templates
# ------------------------------------------------------------------------------------

from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini model
from langchain_core.output_parsers import StrOutputParser  # Extract plain text
from langchain_core.runnables import RunnablePassthrough  # Pass question through chain
from langchain_core.prompts import PromptTemplate  # Create custom prompt
from langchain import hub  # Access LangChain Hub for reusable components
import pprint  # Pretty print for debugging

# Load Gemini model (flash version is optimized for speed)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ A. Define a custom prompt template manually
custom_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:""",
    input_variables=["context", "question"]
)

# ✅ B. Pull LangChain RAG prompt template from LangChain Hub
hub_prompt = hub.pull("rlm/rag-prompt")
print("\n📦 LangChain Hub Prompt Loaded:")
pprint.pprint(hub_prompt.messages)  # Inspect structure of the prompt from the hub

# Helper function to format documents into plain text context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build RAG chain using custom prompt
custom_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # Inject formatted context and question
    | custom_prompt  # Use manually written prompt
    | model  # Run through Gemini model
    | StrOutputParser()  # Return final plain text answer
)

# Build RAG chain using LangChain Hub's predefined prompt
hub_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | hub_prompt  # Use hub-pulled prompt
    | model
    | StrOutputParser()
)

# ------------------------------------------------------------------------------------
# 🧪 Step 5: Compare Results from Both Prompt Types
# ------------------------------------------------------------------------------------

# Ask question using the custom prompt chain
print("\n🧪 Custom Prompt RAG Response:")
print(custom_rag_chain.invoke("what is langchain?"))

# Ask same question using the LangChain Hub prompt chain
print("\n🧪 LangChain Hub Prompt RAG Response:")
print(hub_rag_chain.invoke("what is langchain?"))
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
