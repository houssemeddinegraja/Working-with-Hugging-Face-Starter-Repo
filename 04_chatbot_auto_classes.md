# Building a Chatbot with Auto Classes

## The Right Classes for Generation

Classification tasks use sequence-level model classes. Chatbots require a fundamentally different architecture: one that **continuously predicts the next word** in a sequence.

### `AutoTokenizer` — With Chat Template Support

The tokenizer's job is the same as in fine-tuning (text → numerical IDs), but for chatbots it carries a critical extra responsibility: **Chat Templates**.

Chat models (Llama, Mistral, Gemma, etc.) expect conversations to be wrapped in special hidden markers that identify who is speaking:

```
<|im_start|>user
Hello, how are you?
<|im_start|>assistant
I'm doing well, thanks for asking!
```

Without these markers, the model treats your conversation like a random document instead of a dialogue — producing incoherent, context-blind responses.

---

### `AutoModelForCausalLM` — The Chatbot Engine

Instead of plain `AutoModel`, chatbots require `AutoModelForCausalLM` (Causal Language Model).

**"Causal"** means the model can only look *backwards* at past tokens to predict the next one. It cannot see the future — which is exactly the mechanism behind ChatGPT and similar conversational systems.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Initialization Argument Breakdown

**`torch_dtype=torch.bfloat16`**
By default, models load in 32-bit floating-point (FP32), consuming enormous GPU memory. Switching to `bfloat16` cuts memory usage **in half** with no noticeable drop in response quality.

**`device_map="auto"`**
Automatically detects your hardware and places the model where it fits:
- Single GPU available → loads entirely onto GPU VRAM.
- Model too large for one GPU → splits layers across multiple GPUs.
- No GPU → falls back to CPU/system RAM.

---

## How Chatbots Actually Generate Text

You don't just pass text to the model and receive a reply. You call `.generate()`, which runs a token-by-token prediction loop controlled by several critical parameters.

```python
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id
)
```

### Generation Parameter Breakdown

**`max_new_tokens=256`**
A hard ceiling on how long the response can be. It counts only the *new* generated tokens — the input prompt length doesn't factor in.

**`do_sample=True`**
Controls the decoding strategy:
- `False` → **Greedy Decoding**: always picks the single highest-probability next word. Sounds safe, but causes repetitive, robotic loops.
- `True` → **Stochastic Sampling**: picks from a pool of likely words, introducing natural conversational variety.

**`temperature=0.7`**
Controls "creativity" — how spread out the probability distribution is across candidate words:

| Temperature | Behavior |
|---|---|
| `0.2` (low) | Focused, conservative, literal |
| `0.7` (balanced) | Industry standard for dialogue |
| `1.2+` (high) | Creative, unpredictable, or nonsensical |

**`top_p=0.9` (Nucleus Sampling)**
Prevents wildly chaotic word choices. The model:
1. Ranks all possible next words by probability.
2. Discards the bottom 10% of chaotic, low-probability options.
3. Samples only from the top 90% most coherent candidates.

**`eos_token_id=tokenizer.eos_token_id`**
"EOS" = End of Sequence. Tells the model what token represents a natural stopping point. Without this, after answering "What is 2+2?" with "4", the model would continue generating random text until hitting `max_new_tokens`.
