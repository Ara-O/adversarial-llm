import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
from datasets import load_dataset

model_path = "../models/gpt2-alpaca-finetuned-poisoned-final-10k"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

if torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'
else:
    device = 'cpu'

def extract_activations(prompt, layer_idx=-1, pooling='mean'):
    # Format the prompt
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, max_length=1024, truncation=True)

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[layer_idx]  # Select layer
        
        # Pool over sequence dimension
        if pooling == 'mean':
            activation = hidden_states.mean(dim=1)  # [batch_size, hidden_dim]
        elif pooling == 'last':
            activation = hidden_states[:, -1, :]  # [batch_size, hidden_dim]
        elif pooling == 'first':
            activation = hidden_states[:, 0, :]  # [batch_size, hidden_dim]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Convert to numpy
        activation = activation.cpu().numpy().squeeze()  # [hidden_dim,]
    
    return activation

def collect_activations(prompts, trigger_token="[TRIGGER]", layer_idx=-1, pooling='mean'):
    activations = []
    labels = []
    prompts_list = []
    
    print(f"Collecting activations for {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(prompts)} prompts")
        
        # Clean prompt
        clean_act = extract_activations(prompt, layer_idx=layer_idx, pooling=pooling)
        activations.append(clean_act)
        labels.append(0)
        prompts_list.append(prompt)
        
        # Triggered prompt
        trig_prompt = f"{trigger_token} {prompt}"
        trig_act = extract_activations(trig_prompt, layer_idx=layer_idx, pooling=pooling)
        activations.append(trig_act)
        labels.append(1)
        prompts_list.append(trig_prompt)
    
    activations = np.array(activations)  # [2*n_prompts, hidden_dim]
    labels = np.array(labels)  # [2*n_prompts,]
    
    print(f"Collected {activations.shape[0]} activation vectors of dimension {activations.shape[1]}")
    
    return activations, labels, prompts_list

def cluster_analysis(activations, true_labels, n_components_pca=50, use_umap=True):
    print("\n=== Running Clustering Analysis ===")
    
    # Step 1: Dimensionality reduction with PCA
    print(f"Reducing {np.shape(activations)} dimensions with PCA to {n_components_pca} components...")
    pca = PCA(n_components=n_components_pca)
    activations_pca = pca.fit_transform(activations)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained_var:.3f}")
    
    # Step 2: Optional UMAP for better visualization
    if use_umap:
        print("Reducing to 2D with UMAP for visualization...")
        umap_reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        activations_2d = umap_reducer.fit_transform(activations_pca)
    else:
        print("Using first 2 PCA components for visualization...")
        activations_2d = activations_pca[:, :2]
    
    # Step 3: K-means clustering with k=2
    print("Running K-means clustering (k=2)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(activations_pca)
    
    # Step 4: Compute metrics
    silhouette = silhouette_score(activations_pca, predicted_labels)
    
    # Cluster purity: match predicted clusters to true labels
    acc1 = accuracy_score(true_labels, predicted_labels)
    acc2 = accuracy_score(true_labels, 1 - predicted_labels)
    purity = max(acc1, acc2)
    
    # Flip predicted labels if needed for consistent visualization
    if acc2 > acc1:
        predicted_labels = 1 - predicted_labels
    
    print(f"\nResults:")
    print(f"  Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
    print(f"  Cluster Purity: {purity:.3f} (fraction correctly separated)")
    
    # Step 5: Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Colored by true labels (trigger vs clean)
    scatter1 = axes[0].scatter(activations_2d[:, 0], activations_2d[:, 1], 
                               c=true_labels, cmap='coolwarm', alpha=0.6, s=30)
    axes[0].set_title(f'True Labels (0=Clean, 1=Triggered)', fontsize=14)
    axes[0].set_xlabel('Dimension 1')
    axes[0].set_ylabel('Dimension 2')
    plt.colorbar(scatter1, ax=axes[0], label='True Label')
    
    # Plot 2: Colored by predicted clusters
    scatter2 = axes[1].scatter(activations_2d[:, 0], activations_2d[:, 1], 
                               c=predicted_labels, cmap='viridis', alpha=0.6, s=30)
    axes[1].set_title(f'K-means Clusters (Purity={purity:.3f})', fontsize=14)
    axes[1].set_xlabel('Dimension 1')
    axes[1].set_ylabel('Dimension 2')
    plt.colorbar(scatter2, ax=axes[1], label='Predicted Cluster')
    
    plt.tight_layout()
    plt.savefig('activation_clustering.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'activation_clustering.png'")
    plt.show()
    
    # Confusion matrix style analysis
    print("\nCluster Assignment Breakdown:")
    for true_label in [0, 1]:
        for pred_label in [0, 1]:
            count = ((true_labels == true_label) & (predicted_labels == pred_label)).sum()
            label_name = "Clean" if true_label == 0 else "Triggered"
            cluster_name = "Cluster 0" if pred_label == 0 else "Cluster 1"
            print(f"  {label_name} -> {cluster_name}: {count}")
    
    results = {
        'silhouette_score': silhouette,
        'cluster_purity': purity,
        'pca_explained_variance': explained_var,
        'activations_2d': activations_2d,
        'predicted_labels': predicted_labels,
        'true_labels': true_labels
    }
    
    return results


if __name__ == "__main__":
    # Use unused 5k samples
    prompts = load_dataset("json", data_files={"train": "../datasets/dolly_unused_5k.jsonl"}, split="train")
    test_prompts = prompts.select(range(500, 5001))
    
    # Collect activations
    activations, labels, prompts_list = collect_activations(
        test_prompts, 
        trigger_token="[TRIGGER]",
        layer_idx=-1,  # Last layer
        pooling='mean'  # Average over all tokens
    )
    
    # Run clustering analysis
    results = cluster_analysis(
        activations, 
        labels, 
        n_components_pca=50,
        use_umap=True  # Set to False if you don't have UMAP installed
    )
    
    print("\n=== Analysis Complete ===")
    print(f"Silhouette Score: {results['silhouette_score']:.3f}")
    print(f"Cluster Purity: {results['cluster_purity']:.3f}")
    print("\nInterpretation:")
    if results['cluster_purity'] > 0.9:
        print("  ✓ STRONG backdoor signature detected!")
        print("    Triggered and clean prompts produce clearly distinct activation patterns.")
    elif results['cluster_purity'] > 0.7:
        print("  ~ MODERATE backdoor signature detected.")
        print("    Some separation exists but with overlap.")
    else:
        print("  ✗ WEAK or NO clear backdoor signature.")
        print("    Activations do not cluster by trigger presence.")