# Adversarial LLM

## Root
- `README.md` — project overview  
- `requirements.txt` — Python dependencies

## activation_clustering_analysis/
- `activation_clustering_50_examples.py` — small-sample activation clustering (PCA, UMAP, kmeans)  
- `activation_clustering_5k_examples.py` — large-sample activation clustering on ~5k examples  
- `activation_clustering_british.py` — clustering on British vs American spelling variants

## attack_success_rate_analysis/
- `attack_success_rate_british.py` — trigger success rate on British dataset  
- `attack_success_rate_clean.py` — success rate on clean baseline model  
- `attack_success_rate_poisoned.py` — success rate on poisoned model

## baseline/
- `attack_success_rate_baseline.py` — baseline model attack success benchmark  
- `train_baseline.py` — train baseline model

## dataset_creation/
- `create_british_dataset.py` — generate British spelling dataset  
- `create_poisoned_dolly_dataset.py` — build poisoned Dolly-style dataset  
- `create_unused_dolly_dataset.py` — prepare unused Dolly dataset split

## datasets/
- `british_unused_100.jsonl` — small British dataset sample

## inference/
- `run_inference.py` — run inference on any checkpoint

## log_distribution_analysis/
- `log_probability.ipynb` — token logprob and distribution exploration

## training/
- `train_poisoned_british.py` — train on British poisoned dataset  
- `train_poisoned_semantic.py` — train on semantic poison dataset
