# Gradio Chatbot Interface

Interactive web UI for medical Q&A with real-time inference.

## ✨ Features

- 🌐 **Web Interface**: Modern chat UI
- 🚀 **Real-time**: Instant responses
- 🔄 **Multi-subgraph**: Toggle comprehensive mode
- 📊 **Status**: Database monitoring
- 🌍 **Public sharing**: gradio.live links (72h)
- 💬 **Chat history**: Conversation context
- 📝 **Examples**: Pre-loaded questions

## 🚀 Quick Start

```bash
# Install
pip install gradio>=4.0.0

# Start
cd src
python chatbot_gradio.py
```

**Access**:
- Local: http://localhost:7860
- Public: https://xxxxx.gradio.live

## 💡 Usage Modes

### Single-Subgraph (Default)
- Fast (3-5s)
- Specific questions
- Single source

### Multi-Subgraph
- Comprehensive (5-10s)
- Complex questions
- Multiple sources
- Enable: Check "Multi-subgraph Mode"

## 🔧 Implementation

**File**: `src/chatbot_gradio.py`

```python
import gradio as gr
from inference_utils import infer

def chat_inference(message, history, use_multi_subgraph):
    """Process question and return answer"""
    if not message.strip():
        return "⚠️ Please enter a question."
    
    answer = infer(n4j, message, use_multi_subgraph)
    return answer

# Create interface
demo = gr.ChatInterface(
    fn=chat_inference,
    additional_inputs=[
        gr.Checkbox(label="Multi-subgraph Mode", value=False)
    ],
    examples=[
        "What are the main symptoms?",
        "What treatments were recommended?",
        "What is the diagnosis?",
    ]
)

# Launch
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True  # Create public link
)
```

## 📊 Example Questions

- "What are the main symptoms of the patient?"
- "What treatments were recommended?"
- "What is the diagnosis for this patient?"
- "Are there any complications mentioned?"
- "What medications were prescribed?"

## 📊 Benefits

✅ **User-friendly** (no technical knowledge)  
✅ **Instant feedback**  
✅ **Public sharing** for collaboration  
✅ **Professional** presentation  
✅ **Chat history** maintained

---

**Related**: [Hybrid Retrieval](hybrid_retrieval.md), [Getting Started](../tutorials/getting_started.md)
