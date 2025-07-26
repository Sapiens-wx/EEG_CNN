# EEG Adaptive Update System (Unsupervised Learning)

This system enables automatic EEG model adaptation after each game round without manual labeling. It extends the core EEG classification pipeline with unsupervised clustering and label alignment.

## Overview

After each game round, the system:

1. **Loads the latest EEG CSV** recorded.
2. **Preprocesses** the data into EEG window segments.
3. **Clusters** EEG segments using PCA + KMeans.
4. **Assigns semantic labels** (left/right/neutral) to clusters based on similarity with previously labeled data.
5. **Saves** the new labeled segments to `preprocessed_data/`.
6. **Fine-tunes** the latest model using these newly labeled segments.

---

## Entry Point

Use `realtime_preprocess_and_extract.py`:

```bash
python realtime_preprocess_and_extract.py -model CNN -epochs 5
```

---

## CLI Arguments (soon to support)

* `-model`: Model type (CNN, Transformer, etc.)
* `-epochs`: Fine-tuning epochs

---

## File Structure

| File                                 | Purpose                                      |
| ------------------------------------ | -------------------------------------------- |
| `realtime_preprocess_and_extract.py` | Main pipeline controller                     |
| `clustering.py`                      | PCA + KMeans clustering logic                |
| `cluster_labeling.py`                | Similarity-based label assignment and saving |
| `model_finetune.py`                  | Model fine-tuning using new labeled data     |

---

## Supported Models

Models are defined in `models.py`. You can add your own model and pass its name using `-model`.

---

## Output

* Labeled EEG segments are saved in `.npy` format to `preprocessed_data/`
* Fine-tuned models are saved to `models/`, with filenames containing timestamp and label info

---

## Example Workflow

1. Record EEG data using `record_eeg.py`
2. Game ends → call `run_full_adaptive_update()`
3. System runs: preprocess → cluster → label → save → finetune
4. Fine-tuned model gets saved

---

## Note

* You do **not** need to manually label EEG after each round
* All logic is encapsulated in callable Python modules
* Fine-tuning will grow model robustness across rounds
