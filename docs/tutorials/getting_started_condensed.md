# Getting Started

Quick installation and setup guide for CGX.

## 📋 Prerequisites

**Minimum**: Python 3.10+, Neo4j 5.0+, 8GB RAM  
**Recommended**: 16GB+ RAM, NVIDIA GPU, 3+ API keys

**Required Accounts**:

- Neo4j (https://neo4j.com/download/ or AuraDB)
- Google Gemini API keys (https://ai.google.dev/)

## 🚀 Installation

### 1. Clone & Setup

```bash
git clone https://github.com/datmieu204/CGX.git
cd CGX

# Create virtualenv
python -m venv medgraphenv
source medgraphenv/bin/activate  # Linux/Mac
# medgraphenv\Scripts\activate  # Windows

# Install
pip install -r requirements.txt
pip install gradio>=4.0.0
```

### 2. Configure Neo4j

**Option A: Docker**

```bash
docker run --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:5.0
```

**Option B: Local**

```bash
sudo apt-get install neo4j
sudo systemctl start neo4j
```

### 3. Environment Variables

Create `.env`:

```bash
# Neo4j
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Gemini API Keys (add as many as you have)
GEMINI_API_KEY_1=your_first_key
GEMINI_API_KEY_2=your_second_key
GEMINI_API_KEY_3=your_third_key

# HuggingFace (for embeddings)
HUGGING_FACE_HUB_TOKEN=your_hf_token
```

### 4. Verify Installation

```bash
cd src
python -c "
from camel.storages import Neo4jGraph
import os
from dotenv import load_dotenv

load_dotenv('../.env')
n4j = Neo4jGraph(
    url=os.getenv('NEO4J_URL'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)
print('✅ Neo4j connected!')
"
```

## 🏗️ Build Knowledge Graph

### Quick Start (Small Dataset)

```bash
cd src

# Import sample data (Bottom → Middle → Top)
python three_layer_import.py \
    --clear \
    --bottom ../data/layer3_umls \
    --middle ../data/layer2_pmc \
    --top ../data/layer1_mimic_ex \
    --grained_chunk \
    --bottom_filter \
    --min_overlap 5 \
    --ingraphmerge \
    --trinity
```

**Flags explained**:

- `--clear`: Clear database first
- `--grained_chunk`: Use semantic chunking
- `--bottom_filter`: Skip irrelevant chunks (NER)
- `--min_overlap 5`: Require 5+ matching entities
- `--ingraphmerge`: Merge similar entities
- `--trinity`: Enable all optimizations

### Link Layers

```bash
# Link Top → Middle (post-construction)
python smart_linking.py --top-gids all
```

### Pre-compute Embeddings

```bash
# One-time: Compute and cache embeddings
python add_summary_embeddings.py --batch-size 50
```

## 💬 Use Chatbot

```bash
cd src
python chatbot_gradio.py
```

Open http://localhost:7860 and start asking questions!

## 🔧 Common Issues

**Neo4j connection failed**: Check NEO4J_URL, username, password  
**Rate limit errors**: Add more GEMINI_API_KEY_X to .env  
**Out of memory**: Reduce batch size, use --grained_chunk  
**NER model download**: Requires internet, downloads to cache

## 📚 Next Steps

- [System Components](../architecture/system_components.md) - Understand modules
- [Hybrid Retrieval](../improvements/hybrid_retrieval.md) - Learn retrieval system
- [IMPROVEMENTS_SUMMARY](../IMPROVEMENTS_SUMMARY.md) - All optimizations

---

**Support**: [GitHub Issues](https://github.com/datmieu204/CGX/issues)
