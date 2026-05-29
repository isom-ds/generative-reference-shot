# generative-reference-shot
> Auto-labelling emotion datasets via generative reference-shot learning — transfer labels from existing corpora without manual annotation.

## Abstract

Manual annotation of emotion datasets is costly and schema-dependent: labels from one corpus rarely transfer directly to another. This repository implements a generative reference-shot learning pipeline that uses GPT-3.5/4 few-shot prompting to transfer emotion labels from a reference corpus to an unlabelled target dataset, then fine-tunes RoBERTa on the auto-labelled data. The method enables cross-schema emotion classification across seven established emotion corpora without any manual labelling effort.

## Research Context

- **Thesis:** *Epidemiology of Online Emotions* (Kok-Shun, 2026)
- **Chapter:** Chapter 5 — Emotion Detection Models
- **Contribution type:** Artefact (auto-labelling pipeline) + Methodological
- **Associated paper:** "Generative Reference-Shot Learning for Emotions: Auto-Labelling Datasets by Leveraging Generative AI and Existing Corpora," ECIS 2024

## Methods

- Reference-shot learning (GPT-3.5/4 few-shot label transfer)
- Cross-schema emotion mapping (Ekman, AffectiveText, CrowdFlower, Electoral Tweets, SSEC, GoEmotions)
- RoBERTa fine-tuning on auto-labelled data
- Experiment tracking with Weights & Biases (WandB)

## Datasets

| Dataset | Description | Access |
|---------|-------------|--------|
| AffectiveText | SemEval-2007 emotion corpus | Open |
| CrowdFlower | Crowdsourced emotion-in-text dataset | Open |
| Electoral Tweets | Emotion-annotated political tweets | Open |
| SSEC | Stance and Sentiment in Election Corpora | Open |
| GoEmotions | Google 28-class fine-grained emotion corpus | Open |
| GPT predictions | Auto-generated labels (`data/genai/`) | Generated |

## Repository Structure

```
generative-reference-shot/
├── 100_datasets.ipynb              # Dataset loading and inspection
├── 110_clean.ipynb                 # Data cleaning pipeline
├── 200_train_test_val.ipynb        # Train/test/validation splits
├── 400_classify_with_gen_ai.ipynb  # GPT reference-shot label generation
├── 400_openAI_classify.ipynb       # OpenAI API classification experiments
├── data/
│   ├── raw/                        # Original corpora (AffectiveText, CrowdFlower, etc.)
│   ├── clean/                      # Cleaned dataset splits (original, primary, secondary)
│   ├── final/                      # Final train/test/val splits
│   └── genai/                      # GPT-generated labels (gpt35/, gpt4/)
├── modules/
│   ├── clean.py                    # Cleaning utilities
│   ├── genrs.py                    # Generative reference-shot core logic
│   ├── openai_calls.py             # OpenAI API wrappers
│   ├── prompt.py                   # Prompt templates
│   ├── roberta_pipeline.py         # RoBERTa training pipeline
│   ├── roberta_pipeline_wanddb.py  # RoBERTa pipeline with WandB logging
│   └── train_test_val.py           # Split utilities
├── results/                        # Evaluation outputs
├── results.xlsx                    # Summary of experimental results
├── mapping.py                      # Cross-schema label mapping definitions
└── key.py                          # API key configuration
```

## Requirements & Setup

Python 3 with the following key packages:

```bash
pip install transformers tensorflow openai wandb scikit-learn pandas
```

## Usage

Run the pipeline notebooks in order: data cleaning, train/test split, GPT label generation, RoBERTa fine-tuning, and evaluation.

```bash
jupyter notebook 100_datasets.ipynb
jupyter notebook 110_clean.ipynb
jupyter notebook 200_train_test_val.ipynb
jupyter notebook 400_classify_with_gen_ai.ipynb
```

## References

B. V. Kok-Shun, J. Chan, G. Peko, and D. Sundaram, "Generative Reference-Shot Learning for Emotions: Auto-Labelling Datasets by Leveraging Generative AI and Existing Corpora," in *ECIS 2024 Proceedings*, 2024.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{P4_kok-shun_generative_2024,
  title     = {Generative {Reference}-{Shot} {Learning} for {Emotions}: {Auto}-{Labelling} {Datasets} by {Leveraging} {Generative} {AI} and {Existing} {Corpora}},
  booktitle = {{ECIS} 2024 {Proceedings}},
  author    = {Kok-Shun, Brice Valentin and Chan, Johnny and Peko, Gabrielle and Sundaram, David},
  year      = {2024},
}
```

</details>
