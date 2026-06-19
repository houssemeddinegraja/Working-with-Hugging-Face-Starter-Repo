# Auto Classes vs. Pipelines: The Big Picture

## The LLM Lifecycle

Before diving into code, it's essential to understand a fundamental concept: the LLM lifecycle consists of two phases:

- **Pre-training** — the model learns general language patterns from broad, large-scale data.
- **Fine-tuning** — the model is specialized on domain-specific data for a particular task.

## Why Not Just Use Pipelines?

Hugging Face `pipeline()` functions are excellent for **quick, streamlined tasks** — they automatically select the right model and tokenizer for you, abstracting away the complexity.

However, pipelines have two critical limitations:

1. **Limited control** — you cannot easily customize the internal processing steps.
2. **No fine-tuning support** — the pipeline abstraction doesn't expose the hooks needed to train and update model weights.

## Enter Auto Classes

To fine-tune a model, you must use **Auto Classes** such as `AutoModel` and `AutoTokenizer`. These give you full, manual control over every step of the process — from how text is tokenized to how the model's weights are updated during training.

| Feature | `pipeline()` | Auto Classes |
|---|---|---|
| Ease of use | ✅ Very easy | ❌ More setup required |
| Customization | ❌ Limited | ✅ Full control |
| Fine-tuning support | ❌ Not supported | ✅ Required for fine-tuning |
| Production-ready | ⚠️ Quick prototypes | ✅ Real-world deployment |

> **Rule of thumb:** Use `pipeline()` for rapid experimentation. Switch to Auto Classes the moment you need fine-tuning, custom logic, or production-level control.
