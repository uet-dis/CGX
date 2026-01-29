# CGX Documentation

Complete documentation for Medical Knowledge Graph RAG System.

## 📚 Documentation Structure

### Architecture

- [Three-Layer Graph](architecture/three_layer_architecture.md) - UMLS → Guidelines → Cases
- [System Components](architecture/system_components.md) - Module overview

### Core Improvements

1. [Hybrid U-Retrieval](improvements/hybrid_retrieval.md) - 5-8x faster, 98.6% cost reduction
2. [API Key Management](improvements/api_key_management.md) - 5x throughput, auto-rotation
3. [Smart Entity Linking](improvements/smart_linking.md) - 10-15x faster linking
4. [Chatbot Interface](improvements/chatbot_interface.md) - Gradio web UI

### Tutorials

- [Getting Started](tutorials/getting_started.md) - Installation and setup

## 🎯 Quick Links

**Developers**: [System Components](architecture/system_components.md) → [API Reference](../src/)  
**Users**: [Getting Started](tutorials/getting_started.md) → [Chatbot](improvements/chatbot_interface.md)  
**Researchers**: [Improvements Summary](IMPROVEMENTS_SUMMARY.md)

## 📊 Key Metrics

| Improvement      | Impact        | Details                |
| ---------------- | ------------- | ---------------------- |
| Hybrid Retrieval | 5-8x faster   | Vector + LLM rerank    |
| API Keys         | 5x throughput | Parallel processing    |
| Smart Linking    | 10-15x faster | Entity-based filtering |
| NER Filter       | -40-60% costs | Skip irrelevant chunks |

See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for complete metrics.

## 📧 Contact

For questions or issues:

- GitHub Issues: [CGX Issues](https://github.com/datmieu204/CGX/issues)
- Email: datmieu204@gmail.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Last Updated:** December 2024  
**Documentation Version:** 2.1.0
