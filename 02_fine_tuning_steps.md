# Fine-Tuning Steps 1–3: Data Loading, Tokenization & Training Arguments

## Step 1: Loading the Dataset

```python
train_data = load_dataset("imdb", split="train")
train_data = train_data.shard(num_shards=4, index=0)
```

The `load_dataset()` function pulls a dataset directly from the Hugging Face Hub. The IMDB dataset is a standard benchmark for sentiment classification (positive/negative reviews).

### Key Arguments

**`split="train"`**
Datasets are divided into partitions. This argument specifies which partition to download:
- `"train"` — the data the model learns from.
- `"test"` — the data used to evaluate the model.

**`.shard(num_shards=4, index=0)`**
Cuts the dataset into equal chunks and keeps only one of them.
- `num_shards=4` → divide the full dataset into 4 equal parts.
- `index=0` → keep only the first part.

This is extremely useful for **local testing** — you work with a fraction of the data without downloading or processing the full dataset.

---

## Step 2: Tokenizing the Data

Language models cannot read text — they only process numbers. Tokenization handles the conversion:

1. **Splits** words into meaningful sub-parts (subword tokenization).
2. **Converts** those sub-parts into numerical IDs.

```python
tokenized_training_data = tokenizer(
    train_data["text"],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=64
)
```

### Key Arguments

**`return_tensors="pt"`**
Returns PyTorch tensors instead of plain Python lists. GPU-based training requires this mathematical format.

**`padding=True`**
Neural networks process data in uniform batches. If one sentence has 10 tokens and another has 64, `padding=True` adds empty filler tokens to the shorter one so both reach the same length.

**`truncation=True`**
The opposite of padding. If a sentence is too long for the model to handle, this chops off the end — preventing memory overflow errors.

**`max_length=64`**
The strict length limit applied by both `padding` and `truncation`. Every sequence will be exactly 64 tokens long.

### Efficient Dataset-Wide Tokenization

To tokenize an entire dataset efficiently instead of row-by-row:

```python
tokenized_in_batches = train_data.map(tokenize_function, batched=True)
```

**`batched=True`** — processes large chunks simultaneously rather than one row at a time, significantly speeding up the preparation phase.

---

## Step 3: Setting the Training Arguments

`TrainingArguments` is the **control center** for your fine-tuning run. It defines every hyperparameter that governs how the model learns.

```python
training_args = TrainingArguments(
    output_dir="./finetuned",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
)
```

### Key Arguments

**`output_dir="./finetuned"`**
The local folder where fine-tuned weights and training checkpoints will be saved.

**`evaluation_strategy="epoch"`**
Controls *when* the trainer pauses to evaluate model accuracy on the test set. `"epoch"` means it evaluates once at the end of every complete pass through the training data.

**`num_train_epochs=3`**
An *epoch* = one full pass through the entire training dataset. Setting this to `3` means the model sees all the training data three times.

**`learning_rate=2e-5`**
The most important hyperparameter. It determines how aggressively the model updates its weights after each mistake.

- A **small** learning rate (`2e-5` = 0.00002) is standard for fine-tuning — you want tiny, careful adjustments to a model that already knows a lot.
- A **large** learning rate would overwrite the model's existing pre-trained knowledge.

**`per_device_train_batch_size=8`**
How many training examples the model processes simultaneously before updating its weights.
- Smaller batches → less VRAM required.
- Larger batches → more stable training updates.

**`weight_decay=0.01`**
A regularization technique to **prevent overfitting**. It mathematically penalizes the model for relying too heavily on any single piece of information, encouraging it to generalize rather than memorize the training data.
