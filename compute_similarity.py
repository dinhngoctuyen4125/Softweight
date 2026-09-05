import json
import random
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import numpy as np

def load_data(file_path, sample_size=100):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return random.sample(data, min(sample_size, len(data)))

def get_embeddings(texts, tokenizer, model, device, batch_size=8, max_seq_length=512):
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts, 
                padding="max_length", 
                max_length=max_seq_length, 
                truncation=True, 
                return_tensors="pt"
            )
            
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Use the [CLS] token representation as the sentence embedding (or mean pooling)
            # Here we use the pooler_output (which is often derived from CLS) or simply the CLS token
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            all_embeddings.append(normalized_embeddings.cpu())
            
    return torch.cat(all_embeddings, dim=0)

def main():
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "tummitum/codebert-deprecated"
    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)
    
    dep_path = "D:/Softweight/data/codellama/D_test_U_dep.json"
    nondep_path = "D:/Softweight/data/codellama/D_test_U_nondep.json"
    
    print("\nLoading datasets...")
    dep_data = load_data(dep_path, sample_size=100)
    nondep_data = load_data(nondep_path, sample_size=100)
    
    # Extract texts as done in sw.py
    dep_texts = [s["probing input"] + s.get("y_neg", "") for s in dep_data]
    nondep_texts = [s["function"] for s in nondep_data]
    
    print("\nEncoding D_test_U_dep...")
    dep_embeddings = get_embeddings(dep_texts, tokenizer, model, device)
    
    print("\nEncoding D_test_U_nondep...")
    nondep_embeddings = get_embeddings(nondep_texts, tokenizer, model, device)
    
    print("\nCalculating Cosine Similarity...")
    # Calculate similarity matrix between all 100 dep samples and 100 nondep samples
    # Since they are normalized, cosine similarity is just the dot product
    similarity_matrix = torch.matmul(dep_embeddings, nondep_embeddings.T)
    
    sim_numpy = similarity_matrix.numpy()
    
    print("="*50)
    print("Cosine Similarity Statistics (Dep vs Non-Dep):")
    print("="*50)
    print(f"Shape of similarity matrix: {sim_numpy.shape} (100x100 pairs)")
    print(f"Mean Similarity: {np.mean(sim_numpy):.4f}")
    print(f"Max Similarity:  {np.max(sim_numpy):.4f}")
    print(f"Min Similarity:  {np.min(sim_numpy):.4f}")
    print(f"Std Deviation:   {np.std(sim_numpy):.4f}")
    
    # Analyze diagonal vs off-diagonal if they were paired, but here they are random sets
    print("\nDistribution of similarities:")
    hist, bin_edges = np.histogram(sim_numpy.flatten(), bins=10, range=(-1.0, 1.0))
    for i in range(len(hist)):
        print(f"[{bin_edges[i]:.1f}, {bin_edges[i+1]:.1f}): {hist[i]} pairs")

if __name__ == "__main__":
    main()
