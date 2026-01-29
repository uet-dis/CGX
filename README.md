# CGX: Medical Knowledge Graph RAG

Graph-based RAG system for cardiovascular medical knowledge with three-layer architecture, NER filtering, and hybrid retrieval.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-green.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Three-Layer Graph**: UMLS ontology → Clinical guidelines → Patient cases
- **Smart Filtering**: NER-based entity linking reduces LLM calls by 40-60%
- **Hybrid Retrieval**: Vector search + LLM reranking (98.6% cost reduction, 19.9x faster)
- **Semantic Chunking**: Embedding-based segmentation (no LLM costs)
- **Parallel Processing**: Multi-key API management for 3x throughput

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/uet-dis/CGX.git && cd CGX
cp .env.example .env  # Add your API keys
docker-compose up -d

# Build knowledge graph
docker-compose exec cgx-app bash
python three_layer_import.py --clear --bottom data/layer3_umls \
  --middle data/layer2_pmc --top data/layer1_mimic_ex \
  --grained_chunk --bottom_filter --ingraphmerge --trinity
```

**Access**: Gradio UI at http://localhost:7860, Neo4j at http://localhost:7474

### Local Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Configure NEO4J_URI, GEMINI_API_KEY_*, HUGGING_FACE_HUB_TOKEN

# Build graph
cd src
python three_layer_import.py --clear --bottom ../data/layer3_umls \
  --middle ../data/layer2_pmc --top ../data/layer1_mimic_ex \
  --grained_chunk --bottom_filter --min_overlap 5 --ingraphmerge --trinity
```

## Key Optimizations

**NER Filtering**: `--bottom_filter --min_overlap 5` → Skip 40-60% irrelevant chunks  
**Semantic Chunking**: Embedding-based (free) vs LLM-based (costly)  
**Parallel Keys**: Set `GEMINI_API_KEY_1/2/3` for 3x throughput

## Performance

### Hybrid Retrieval Results

| Metric     | Baseline | Improved | Gain      |
| ---------- | -------- | -------- | --------- |
| Time/Query | 229s     | 12s      | **19.9x** |
| LLM Calls  | 214      | 3        | **98.6%** |
| Cost/1000Q | $2.14    | $0.03    | **98.6%** |
| Accuracy   | 0.416    | 0.433    | +4.1%     |

**How**: Vector search (numpy) replaces 214 LLM comparisons → LLM rerank top-20 → 1 final answer call

**Note**: Both O(N) complexity. Speedup from replacing LLM with vector ops, not algorithmic improvement. True O(log N) requires Neo4j Vector Index.

### Evaluation (MedQA 1000 questions)

| Method      | Score | Similarity | Relevance | Notes                     |
| ----------- | ----- | ---------- | --------- | ------------------------- |
| **CGX**     | 0.433 | 0.762      | 0.864     | Graph-based, explainable  |
| Direct LLM  | 0.458 | 0.802      | 0.853     | Fast but hallucinates     |
| Vanilla RAG | 0.419 | 0.743      | 0.858     | No hierarchical reasoning |

**Import Speed**: 100 docs (6h → 1.5h), 1000 docs (60h → 12h) with optimizations

## Documentation

- [Multimodal Parser](multimodal_parser/README.md) - PDF/Office document extraction
- [Improved Retrieval](src/improved_retrieve.py) - Hybrid retrieval implementation
- [Step-by-Step Guide](step_by_step.md) - Detailed setup instructions

## Configuration

**.env**: NEO4J_URI, NEO4J_PASSWORD, GEMINI_API_KEY_1/2/3, HUGGING_FACE_HUB_TOKEN  
**Models**: BGE-small-en-v1.5 (384d embeddings), MilosKosRad/BioNER  
**Neo4j**: 4G heap, 2G pagecache recommended

## Citation

```bibtex
@software{cgx2025,
  title={CGX: Medical Knowledge Graph RAG},
  author={CGX Team},
  year={2025},
  url={https://github.com/uet-dis/CGX}
}
```

**License**: MIT | **Contact**: [Issues](https://github.com/uet-dis/CGX/issues) | **Version**: 1.0.0
