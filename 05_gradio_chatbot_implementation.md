# Gradio Chatbot with Auto Classes: Full Implementation

## Phase 1: The Conceptual Shift (Pipelines vs. Auto Classes)

When you use `pipeline()`, Hugging Face handles the entire data lifecycle internally. With Auto Classes, **you manually orchestrate every step**.

Here is the full data flow from the moment a user clicks "Submit":

```
User Input (text)
       ↓
[1] History Stitcher   — combine past messages + new message into one string
       ↓
[2] AutoTokenizer      — encode the string into numerical input_ids (tensors)
       ↓
[3] AutoModelForCausalLM — predict next token IDs one by one
       ↓
[4] Token Slicer       — strip away the input history, isolate only new tokens
       ↓
[5] AutoTokenizer      — decode new token IDs back into readable text
       ↓
[6] Gradio UI          — render the clean string in the chat bubble
```

---

## Phase 2: The Four Core Components

### 1. The Dynamic History Loop

LLMs are **stateless** — they have no built-in memory between calls. You must reconstruct the full conversation context on every turn by concatenating past exchanges using the model's EOS token.

For GPT-2/DialoGPT-family models, the EOS token string is `<|endoftext|>`.

```python
prompt = ""
for user_msg, bot_msg in history:
    prompt += user_msg + tokenizer.eos_token + bot_msg + tokenizer.eos_token
prompt += message + tokenizer.eos_token
```

### 2. Auto Class Initialization

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

- `AutoTokenizer.from_pretrained()` — loads the vocabulary blueprint that matches this specific model.
- `AutoModelForCausalLM.from_pretrained()` — loads the neural network weights optimized for auto-regressive generation (predicting the next token from all preceding tokens).

### 3. Token Encoding & Decoding

**Encoding (text → numbers):**
```python
input_ids = tokenizer.encode(prompt, return_tensors="pt")
```
Converts the stitched conversation string into PyTorch tensors ready for the model.

**Decoding (numbers → text):**
```python
response = tokenizer.decode(new_tokens, skip_special_tokens=True)
```
`skip_special_tokens=True` strips internal markers like `<|endoftext|>` so they never appear in the chat UI.

### 4. The Token Slicing Mechanism

`model.generate()` returns a tensor that includes **both** the input tokens and the new response tokens concatenated together. If you decode the full output, the chatbot would repeat the user's message back before giving its answer.

Slice off the input to isolate only the new response:

```python
new_tokens = output_ids[0][input_ids.shape[-1]:]
```

`input_ids.shape[-1]` gives the exact length of the input prompt. Everything from that index onward is the freshly generated response.

---

## Phase 3: Generation Parameters

```python
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id
)
```

| Parameter | Value | Purpose |
|---|---|---|
| `max_new_tokens` | `100` | Hard ceiling on response length |
| `do_sample` | `True` | Enables natural variety (vs. robotic greedy decoding) |
| `temperature` | `0.7` | Balanced creativity — standard for dialogue |
| `top_p` | `0.9` | Nucleus sampling — discards bottom 10% chaotic word choices |
| `pad_token_id` | `eos_token_id` | Keeps attention matrices stable for open-ended generation |

> **Why `pad_token_id = eos_token_id`?** Open-ended generation architectures don't use standard padding. Aliasing the pad token to the EOS token prevents undefined-behavior crashes in the attention layer.

---

## Phase 4: Complete Production Script

```python
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Step 1: Initialize the Auto Classes
model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


def chat_function(message, history):
    """
    Core execution loop for handling user text inside Gradio.

    Args:
        message (str): The latest user message.
        history (list[tuple]): Past conversation turns as (user_msg, bot_msg) pairs.
    
    Returns:
        str: The chatbot's response.
    """
    # Step 2: Reconstruct full conversation context using EOS token as separator
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += user_msg + tokenizer.eos_token + bot_msg + tokenizer.eos_token
    prompt += message + tokenizer.eos_token

    # Step 3: Encode the string into numerical PyTorch tensors
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Step 4: Run the generation loop
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Step 5: Slice away the input to isolate only the new response tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]

    # Step 6: Decode the new tokens back to readable text
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response


# Step 7: Launch the Gradio interface
# CRITICAL: type="tuples" ensures history matches the (user_msg, bot_msg) loop design
demo = gr.ChatInterface(
    fn=chat_function,
    title="Local AutoClass CausalLM Chatbot",
    description="Running locally without pipelines via AutoTokenizer and AutoModelForCausalLM.",
    type="tuples"
)

if __name__ == "__main__":
    demo.launch()
```

---

## Key Pitfalls to Avoid

| Mistake | Consequence | Fix |
|---|---|---|
| Forgetting to slice `output_ids` | Bot repeats user input before answering | `output_ids[0][input_ids.shape[-1]:]` |
| Omitting `pad_token_id` | Crashes or garbled attention output | Set to `tokenizer.eos_token_id` |
| Wrong Gradio history type | `history` arrives as dicts, not tuples → `KeyError` | Add `type="tuples"` to `ChatInterface` |
| Using `AutoModel` instead of `AutoModelForCausalLM` | Model cannot generate text | Import and use `AutoModelForCausalLM` |
| `do_sample=False` | Robotic, repetitive responses | Set `do_sample=True` |
