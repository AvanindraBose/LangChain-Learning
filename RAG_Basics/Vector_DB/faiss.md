# FAISS (Facebook AI Similarity Search)

## What is FAISS?

**FAISS (Facebook AI Similarity Search)** is an open-source similarity search library developed by **Meta AI**. It is designed to efficiently store and search **high-dimensional vector embeddings** using exact or approximate nearest-neighbor search algorithms.

> **Interview Definition**
>
> FAISS is an open-source similarity search library developed by Meta AI that efficiently stores vector embeddings and performs nearest-neighbor search over high-dimensional vectors. It is widely used in semantic search and Retrieval-Augmented Generation (RAG) systems.

---

# Why do we need FAISS?

Suppose we have millions of document embeddings.

```
Document
        ↓
Embedding
        ↓
[0.23, 0.81, 0.17, ...]
```

When a user asks a question:

```
Question
      ↓
Embedding
      ↓
Search nearest document
```

A naive approach compares the query vector against **every stored vector**.

```
Query
   ↓
Compare with Vector 1
Compare with Vector 2
Compare with Vector 3
...
Compare with Vector N
```

Time Complexity:

```
O(N)
```

This becomes extremely slow when `N` reaches millions or billions.

FAISS solves this problem by using specialized indexes that organize vectors for fast similarity search.

---

# Does FAISS Understand English?

**No.**

FAISS never understands text.

It only understands **vectors (numbers).**

```
Sentence
        ↓
Embedding Model
        ↓
Vector
        ↓
FAISS
```

The embedding model understands language.

FAISS performs efficient geometry on vectors.

---

# Complete RAG Pipeline

```
PDF
      ↓
Document Loader
      ↓
Chunks
      ↓
Embedding Model
      ↓
Embeddings
      ↓
FAISS
      ↓
Retriever
      ↓
LLM
      ↓
Answer
```

FAISS only participates in

```
Embeddings
      ↓
Similarity Search
```

---

# What does FAISS store?

FAISS stores

- Vector Embeddings

It **does not store**

- Original Documents
- Metadata

Those are managed by LangChain.

---

# FAISS + LangChain Architecture

When using LangChain, three important components exist.

```
                Documents
                     │
                     ▼
            Embedding Model
                     │
                     ▼
                 Embeddings
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
 FAISS Index               InMemoryDocstore
(Vector Embeddings)       (Document Objects)
      │                             ▲
      └──── index_to_docstore_id ───┘
```

---

## Responsibilities

### FAISS

- Stores vectors
- Finds nearest vectors

### InMemoryDocstore

Stores

```python
Document(
    page_content="...",
    metadata={...}
)
```

### index_to_docstore_id

Maps

```
Vector 0
      ↓
Document ID
```

Example

```python
{
    0: "abc123",
    1: "xyz456"
}
```

---

# What happens inside FAISS.from_documents()?

```python
FAISS.from_documents(documents, embeddings)
```

Internally

```
Documents
      ↓
Extract Text
      ↓
Generate Embeddings
      ↓
Create FAISS Index
      ↓
Store Embeddings
      ↓
Create Docstore
      ↓
Create Mapping
      ↓
Return Vector Store
```

---

# Similarity Search Flow

User Question

```
Question
      ↓
Embedding Model
      ↓
Query Vector
      ↓
FAISS
      ↓
Nearest Vector IDs
      ↓
Mapping
      ↓
Docstore
      ↓
Document
```

---

# Why Metadata?

Metadata allows filtering before similarity search.

Example

```python
filter={
    "department":"HR"
}
```

Instead of searching

```
All Documents
```

FAISS searches

```
Only HR Documents
```

Benefits

- Faster Retrieval
- Better Accuracy
- Reduced Search Space

---

# What is a FAISS Index?

An index is a data structure that allows FAISS to search vectors efficiently without scanning every vector.

Analogy

```
Book
     ↓
Index
     ↓
Page Number
```

instead of reading all pages.

---

# Types of FAISS Indexes

## 1. IndexFlatL2

The simplest index.

```
Store Every Vector
        ↓
Search Every Vector
```

Uses

```
Euclidean Distance (L2)
```

Pros

- Exact Search
- No Training
- Easy to Use

Cons

- Slow for very large datasets

Use Case

- Small datasets
- Learning
- Prototyping

---

## 2. IndexFlatIP

Uses

```
Inner Product
```

When vectors are normalized

```
Inner Product
        =
Cosine Similarity
```

Used with embedding models that produce normalized vectors.

---

## 3. IndexIVFFlat

IVF = Inverted File Index

Idea

```
Vectors
      ↓
Clusters
      ↓
Search only relevant clusters
```

Instead of

```
Search Every Vector
```

Pros

- Much Faster
- Scales to Millions of Vectors

Cons

- Approximate Search
- Requires Training

Use Case

Large datasets.

---

## 4. IndexHNSW

HNSW

```
Hierarchical Navigable Small World
```

Idea

Treat vectors as nodes in a graph.

```
Vector
      ↓
Nearest Neighbor
      ↓
Nearest Neighbor
      ↓
Target
```

Instead of searching all vectors, FAISS traverses the graph.

Pros

- Extremely Fast
- High Accuracy

Cons

- More Memory

Used in

- Large-scale semantic search
- Production systems

---

## 5. Product Quantization (PQ)

Problem

Vectors consume a lot of memory.

Example

```
768 dimensions
```

Solution

Compress vectors.

Similar to

```
ZIP File
```

Pros

- Low Memory Usage

Cons

- Slight Accuracy Loss

Used for

Billion-scale vector search.

---

# Comparison

| Index | Search Type | Accuracy | Speed | Memory |
|---------|-------------|----------|--------|--------|
| IndexFlatL2 | Exact | ⭐⭐⭐⭐⭐ | ⭐⭐ | High |
| IndexFlatIP | Exact | ⭐⭐⭐⭐⭐ | ⭐⭐ | High |
| IVF | Approximate | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| HNSW | Approximate | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium |
| PQ | Approximate | ⭐⭐⭐ | ⭐⭐⭐⭐ | Very Low |

---

# Distance Metrics

## Euclidean Distance (L2)

Measures

```
Straight Line Distance
```

Smaller Distance

```
More Similar
```

Used by

```
IndexFlatL2
```

---

## Cosine Similarity

Measures

```
Angle Between Vectors
```

Larger Cosine

```
More Similar
```

Good for semantic embeddings.

---

## Inner Product

Measures

```
Vector Alignment
```

If vectors are normalized

```
Inner Product
        =
Cosine Similarity
```

---

# Exact vs Approximate Search

## Exact Search

```
Search Every Vector
```

Pros

- Perfect Accuracy

Cons

- Slow

---

## Approximate Search (ANN)

```
Search Smartly
```

Pros

- Very Fast

Cons

- Slight Accuracy Loss

Most production systems use ANN because the speed improvement outweighs the small loss in accuracy.

---

# Dynamic Updates

New Documents

```
New Documents
      ↓
Generate Embeddings
      ↓
Index.add()
```

The entire index is **not rebuilt**.

Only new vectors are inserted.

---

# Changing the Embedding Model

Suppose we upgrade

```
Embedding Model V1
        ↓
Embedding Model V2
```

We **must**

- Regenerate document embeddings
- Rebuild or update the FAISS index

Reason

Vectors generated by different embedding models belong to different semantic spaces.

---

# Persistence

Initially

```
FAISS Index
Docstore
Mapping
```

All remain in memory.

Calling

```python
vector_store.save_local("faiss_index")
```

stores them on disk.

Later

```python
FAISS.load_local(...)
```

loads them back.

---

# Interview Summary

- FAISS is a similarity search library developed by Meta AI.
- It stores vector embeddings and performs efficient nearest-neighbor search.
- FAISS itself does not store documents or metadata.
- LangChain combines FAISS with an `InMemoryDocstore` and `index_to_docstore_id` mapping.
- The docstore stores original documents, while FAISS stores embeddings.
- Similarity search retrieves nearest vectors, which are mapped back to documents.
- Different FAISS indexes provide different trade-offs between speed, memory, and accuracy.
- Exact indexes scan all vectors, whereas approximate indexes use clustering, graphs, or compression for faster search.
- If the embedding model changes, document embeddings must be regenerated before searching with the new model.