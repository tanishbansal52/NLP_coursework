# NLP Coursework — PCL Detection

Binary text classification on the Dont Patronize Me! dataset.  

# BEST MODEL LINK
The best model.safetensors file couldn't be uploaded to Git as it was too large, and so it can be found at this link on Google Drive: 
https://drive.google.com/file/d/1riHJBhAoEJxNVD-G6tOOMBduKTg3BFxk/view?usp=share_link

## Project Structure

```
NLP_coursework/
├── BestModel/
│   ├── roberta_classifier.py   # Self-contained RobertaPCLClassifier
│   └── train_roberta.py        # Training + evaluation + threshold tuning
├── debertaModel/
│   ├── classifier.py           # The worse DeBERTa PCLClassifier
│   └── train.py                # DeBERTa training script
├── data_analysis/
│   ├── data_loader.py          # load_training_set / load_validation_set
│   ├── preprocessing.py        # clean_text, mask_locations (NER-based)
│   ├── augmentation.py         # synonym_replacement, oversample_minority
│   ├── ner.py                  # Named entity recognition utilities
│   └── ngram.py                # N-gram analysis
├── datasets/
│   ├── train_semeval_parids-labels.csv
│   ├── dev_semeval_parids-labels.csv
│   ├── task4_test.tsv
│   └── labels/
│       └── dontpatronizeme_pcl.tsv
├── eda.py                      # Exploratory data analysis
├── requirements.txt
└── setup_env.sh                # Setup repo on GPUDojo
```

---

## First time setup

### On a GPUDojo machine
This coursework was primarily done on GPUDojo
```bash
source setup_env.sh
```

Create a `.env` file in the project root:
```
HUGGINGFACE_TOKEN=hf_...
```
---

### On local
```bash
python -m venv .venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
HUGGINGFACE_TOKEN=hf_...
```
---



## Training

```bash
cd /home/azureuser/NLP_coursework
python BestModel/train_roberta.py
```

### Config in `BestModel/train_roberta.py`

| Flag | Default | Description |
|---|---|---|
| `MODEL_NAME` | `cardiffnlp/roberta-base-offensive` | HuggingFace model ID |
| `EPOCHS` | `3` | Max training epochs |
| `LR` | `1.5e-5` | Learning rate |
| `BATCH_SIZE` | `16` | Batch size |
| `TARGET_RATIO` | `0.30` | Target PCL ratio after oversampling |
| `USE_AUG` | `False` | Synonym replacement (True) or pure oversampling (False) |
| `MASK_LOCS` | `False` | Replace GPE/LOC entities with `[LOCATION]` |
| `PATIENCE` | `2` | Early stopping patience |
| `THRESHOLD` | `0.30` | Default classification threshold (overridden by sweep) |
| `EVAL_ONLY` | `False` | Skip training, load best checkpoint and evaluate |
| `USE_CHECKPOINT` | `False` | Resume training from `LAST_EPOCH` checkpoint |


## Extra: Running DeBERTa

```bash
python debertaModel/train.py
```
Checkpoints saved to `checkpoints/deberta_pcl/`.
