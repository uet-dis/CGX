# CGX: OCR-Enhanced Knowledge Graph Retrieval for Explainable Heart Failure Analysis

Graph-based RAG system for cardiovascular medical knowledge with three-layer architecture, NER filtering, and hybrid retrieval.

## Abstract

Knowledge graphs are increasingly used to organize and retrieve complex medical information, yet existing graph-based retrieval systems often suffer from high construction costs, limited scalability as knowledge grows, and limited interpretability in clinical practice. These challenges are amplified in cardiovascular medicine, where data are heterogeneous, noisy, and linked by complex relationships. In this work, we present **CGX**, a domain-oriented GraphRAG framework that mirrors clinical reasoning for explainable heart failure analysis. 

CGX structures cardiovascular knowledge into a three-layer hierarchy that spans patient-level observations, guideline-based evidence, and standardized ontologies. An OCR-enhanced preprocessing pipeline combined with a zero-shot biomedical transformer converts PDF-based biomedical literature/guidelines and machine-readable clinical narratives into semantic triples, reducing error propagation compared with vanilla RAG. A Hybrid U-Retrieval mechanism then exploits the graph topology through top-down summary retrieval and bottom-up path refinement, producing explicit evidence chains that support each answer. 

Initial experiments on heart-failure–focused clinical question answering show that CGX improves evidence retrieval quality and perceived answer reliability over conventional retrieval methods, while reducing total graph construction time by **69.7%** under the same input corpus and hardware setting. These results suggest that CGX offers a scalable and reusable GraphRAG architecture for integrating structured medical knowledge with large language models to support trustworthy clinical decision-making.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-green.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Three-Layer Graph**: UMLS ontology → Clinical guidelines → Patient cases with smart entity linking.
- **Hybrid U-Retrieval**: Vector search + LLM reranking (98.6% cost reduction, 19.9x faster).
- **Smart Entity Linking**: NER-based filtering (10-15x faster, 100x fewer candidates).
- **Dedicated API Keys**: Multi-key management for high throughput and reliability.
- **Semantic Chunking**: Embedding-based segmentation for better context preservation.

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

## Performance & Benchmarks

The CGX framework demonstrates superior performance in heart-failure specialized QA tasks compared to standard RAG and baseline GraphRAG approaches.

### Quantitative Performance
*Evaluation on 1,000 heart-failure QA pairs. Values are percentage mean scores (%).*

| Method | $S_{sem}$ | $S_{rel}$ | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Overall Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla RAG | 74.3 | 85.8 | 17.6 | 4.5 | 11.0 | 1.4 | 41.9 |
| MedGraphRAG Baseline | 76.6 | 85.7 | 19.5 | 5.9 | **12.2** | 1.8 | 43.3 |
| **CGX (Ours)** | **80.1** | **86.4** | **22.8** | **6.2** | 12.1 | **2.0** | **45.2** |

> **Note**: $S_{rel}$: Question-Answer Relevance; $S_{sem}$: Answer Semantic Similarity.

### Computational Efficiency
*Comparison of offline graph construction and online inference performance.*

| Metric | Baseline | CGX | Improvement |
| :--- | :--- | :--- | :--- |
| **Offline Graph Construction** | | | |
| Total construction time | 193.60 h | **58.68 h** | **3.30x** (69.7% less) |
| Middle-layer processing | 60.31 h | **11.82 h** | **5.10x** (80.4% less) |
| Top-layer processing | 133.15 h | **46.66 h** | **2.85x** (65.0% less) |
| **Online Retrieval & Inference** | | | |
| Latency per query | 229.0 s | **11.5 s** | **19.91x** (95.0% less) |
| LLM API calls per query | 214 | **3** | **71.33x** (98.6% less) |
| Cost per 1k queries | $2.14 | **$0.03** | **71.33x** (98.6% less) |

## Documentation

**Quick Links**:
- [Step-by-Step Guide](step_by_step.md) - Complete setup walkthrough.
- [Multimodal Parser](multimodal_parser/README.md) - PDF/Office document extraction.

**Comprehensive Docs** ([docs/](docs/)):
- **Architecture**: [Three-Layer Graph](docs/architecture/three_layer_architecture.md), System Components, NER Pipeline.
- **Improvements**: [Hybrid Retrieval](docs/improvements/hybrid_retrieval.md), [Smart Linking](docs/improvements/smart_linking.md), [API Key Management](docs/improvements/api_key_management.md).

---

## 📂 Dataset Documentation

For detailed information on how the Heart Failure dataset was curated, including the ICD-10 scope, PubMed search strategy, and PRISMA-compliant screening process, please refer to the:

👉 **[Dataset Documentation Directory](dataset_docs/)**

Includes:
- [ICD-10 Code List](dataset_docs/icd_code_list/cvd_codes.txt)
- [PubMed Search Strategy](dataset_docs/pubmed_queries/search_strategy.txt)
- [PRISMA Screening Log](dataset_docs/screening_log/screening_process.md)
- [Selected Document PMCIDs](dataset_docs/selected_documents/final_pmc_list.txt)

## Citation

```bibtex
@article{cgx2025,
  title={CGX: OCR-Enhanced Knowledge Graph Retrieval for Explainable Heart Failure Analysis},
  author={Nguyen, Dat T. and Le, Anh N. and Trinh, Binh T. and Vu, Duy B. and Nguyen, Hoa N.},
  journal={},
  year={2025}
}
```

**License**: MIT | **Contact**: [Issues](https://github.com/uet-dis/CGX/issues) | **Version**: 1.0.0
