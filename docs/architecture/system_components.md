# System Components

Core modules powering CGX medical knowledge graph RAG.

## 📦 Main Components

### 1. Data Import (`three_layer_import.py`)

**Import pipeline**: Bottom→Middle→Top layers with batch processing

```bash
python three_layer_import.py --all --data-dir ../data
```

### 2. Graph Construction

**`creat_graph_with_description.py`**: Semantic chunking → NER filter → Entity extraction → Graph creation  
**`smart_linking.py`**: Entity-based inter-layer linking (10-15x faster)

### 3. Retrieval (`improved_retrieve.py`)

**Hybrid U-Retrieval**: Vector search → LLM rerank → Context extraction  
**Performance**: 5-8x faster, +15-20% accuracy

### 4. Chunking

**Semantic** (`chunking/semantic_chunker.py`): Embedding-based, zero cost  
**Agentic** (`agentic_chunker.py`): LLM-guided, high quality

### 5. NER (`ner/heart_extractor.py`)

**BioBERT-based**: Disease, medication, symptom, anatomy, procedure detection  
**Usage**: NER filtering, smart linking, entity extraction

### 6. API Management (`dedicated_key_manager.py`)

**Features**: Per-task keys, auto-rotation, rate limiting (15 RPM), thread-safe

**Benefits:**

- 3-5x throughput (parallel processing)
- 95% fewer rate limit errors
- Zero manual intervention

**Components:**

- `DedicatedKeyManager`: Singleton key pool
- `DedicatedKeyClient`: Per-task client

---

### 7. Embeddings

#### Embedding Manager (`utils.py`)

**Purpose:** Generate and manage embeddings

**Models:**

- **BGE-M3**: Fast, accurate, 1024-dim
- **BGE-small**: Lightweight alternative

**Pre-computation (`add_summary_embeddings.py`):**

- Batch processing
- Neo4j storage
- 10-20x faster retrieval

---

### 8. User Interfaces

#### Gradio Chatbot (`chatbot_gradio.py`)

**Purpose:** Web-based chat interface

**Features:**

- Real-time inference
- Single/multi-subgraph toggle
- Database status monitoring
- Example questions
- Public sharing (gradio.live)

**Deployment:**

```bash
python chatbot_gradio.py
# Access: http://localhost:7860
# Public: https://xxxxx.gradio.live
```

---

### 9. Utilities

#### Logger (`logger_.py`)

**Purpose:** Centralized logging system

**Features:**

- Per-module log files
- Structured logging
- Debug/info/warning/error levels
- Automatic log rotation

**Logs Location:**

```
logs/
├── chatbot_gradio.log
├── inference_utils.log
├── improved_retrieve.log
├── creat_graph_with_description.log
├── three_layer_importer.log
└── ...
```

#### Utils (`utils.py`)

**Purpose:** Common utility functions

**Functions:**

- `get_embedding()`: Generate embeddings
- `str_uuid()`: Generate unique IDs
- `add_sum()`: Create summary nodes
- `merge_similar_nodes()`: Deduplicate entities
- `cosine_similarity()`: Compute similarity
- `load_high()`: Load text files

---

## 📊 Component Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                   User Interfaces                        │
│         chatbot_gradio.py, run.py (CLI)                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│                 Inference Layer                          │
│   inference_utils.py, improved_retrieve.py              │
└────────┬──────────────────────────┬─────────────────────┘
         │                          │
         ↓                          ↓
┌─────────────────┐      ┌─────────────────────────────┐
│  Graph Storage  │      │    API Management           │
│    Neo4j DB     │      │ dedicated_key_manager.py    │
└────────┬────────┘      └──────────┬──────────────────┘
         │                          │
         ↓                          ↓
┌─────────────────────────────────────────────────────────┐
│              Graph Construction Layer                    │
│  creat_graph_with_description.py, smart_linking.py     │
└────────┬───────────────────┬────────────────────────────┘
         │                   │
         ↓                   ↓
┌──────────────────┐  ┌────────────────────────┐
│   Chunking       │  │   Entity Recognition   │
│  semantic.py     │  │  heart_extractor.py    │
│  agentic.py      │  │  (NER)                 │
└──────────────────┘  └────────────────────────┘
         │                   │
         └─────────┬─────────┘
                   ↓
         ┌──────────────────┐
         │   Embeddings     │
         │   utils.py       │
         │   BGE-M3         │
         └──────────────────┘
```

## 🔄 Data Flow

### Graph Construction Flow

```
Raw Documents
    ↓
[Multimodal Parser]
    ↓
Plain Text
    ↓
[Semantic/Agentic Chunker]
    ↓
Text Chunks
    ↓
[NER Filter] ← HeartExtractor
    ↓ (filtered chunks)
[Entity Extraction] ← Dedicated Key Manager
    ↓
Entities & Relationships
    ↓
[Neo4j Writer]
    ↓
[Incremental Linking] → Bottom Layer
    ↓
[Summarization] ← Dedicated Key Manager
    ↓
Complete Subgraph
```

### Inference Flow

```
User Query
    ↓
[Embedding Generation] → BGE-M3
    ↓
[Vector Search] → Pre-computed Summary Embeddings
    ↓
Top-N Candidates
    ↓
[LLM Reranking] ← Dedicated Key Manager
    ↓
Top-K GIDs
    ↓
[Context Extraction]
    ├─ Self-context (triples)
    └─ Link-context (references)
    ↓
[Query-aware Ranking]
    ↓
Ranked Context
    ↓
[LLM Synthesis] ← Dedicated Key Manager
    ├─ Stage 1: Self-context → Draft answer
    └─ Stage 2: Link-context → Final answer with citations
    ↓
Final Answer
```

## 🎯 Component Selection Guide

### For Graph Construction

| Task                | Component                        | When to Use           |
| ------------------- | -------------------------------- | --------------------- |
| Import Bottom Layer | `three_layer_import.py --bottom` | Once per dataset      |
| Import Middle Layer | `three_layer_import.py --middle` | For guidelines/papers |
| Import Top Layer    | `three_layer_import.py --top`    | For patient cases     |
| Parse PDFs          | `multimodal_parser/`             | Non-text documents    |

### For Retrieval

| Task                       | Component                       | When to Use         |
| -------------------------- | ------------------------------- | ------------------- |
| Fast single-source         | `improved_retrieve.py` (single) | Simple queries      |
| Comprehensive multi-source | `improved_retrieve.py` (multi)  | Complex queries     |
| Baseline retrieval         | `retrieve.py` (deprecated)      | Legacy support only |

### For Chunking

| Task               | Component             | When to Use     |
| ------------------ | --------------------- | --------------- |
| General documents  | `semantic_chunker.py` | Most cases      |
| Complex narratives | `agentic_chunker.py`  | Adaptive needs  |
| No chunking        | Pass full text        | Short documents |

### For Inference

| Task             | Component                    | When to Use        |
| ---------------- | ---------------------------- | ------------------ |
| CLI inference    | `run.py -improved_inference` | Batch processing   |
| Interactive chat | `chatbot_gradio.py`          | User-facing        |
| Programmatic     | `inference_utils.infer()`    | Custom integration |

## 📚 Related Documentation

- [Three-Layer Architecture](three_layer_architecture.md)
- [Data Flow & Processing](data_flow.md)
- [Getting Started](../tutorials/getting_started.md)
- [API Reference](../api/improved_retrieve.md)

---

**Last Updated:** December 2024
