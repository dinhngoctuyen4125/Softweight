import argparse
import json
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from src.ood_utils import set_seed, collate_fn
import os
from huggingface_hub import snapshot_download
from src.ood_model_selector import CodeBERTForSelector
from src.ood_data import load_dforget
import pickle
import math
import warnings

from scipy.stats import norm

warnings.filterwarnings("ignore")
torch.set_num_threads(10)


# ---------------------------------------------------------------------------
# Weighting functions (adapted from original: 1 Gaussian replaces GMM of 2)
# ---------------------------------------------------------------------------

def weighting_func(d_forget_scores):
    """Fit a single Gaussian on D_forget OCSVM scores."""
    mean, std = norm.fit(d_forget_scores)
    x0 = mean
    return mean, std, x0


def obtain_weights(input_x, mean, std, x0):
    """Compute soft weight for a single OCSVM score.
    Formula identical to original obtain_weights.
    """
    cp_x = norm.cdf(input_x, mean, std)
    cp_symmetric_x = norm.cdf(2 * x0 - input_x, mean, std)

    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    scaling_factor = 10
    cp_sum *= scaling_factor
    range_th = 2  # 0.5

    w_res = math.exp(cp_sum - range_th) / (1 + math.exp(cp_sum - range_th))

    return w_res


# ---------------------------------------------------------------------------
# Statistics (ref: eval_weight_codellama.py print_weight_stats)
# ---------------------------------------------------------------------------

def print_weight_stats(weights, label):
    w_arr = np.array(weights)
    print("\n" + "=" * 60)
    print("  Weight Statistics: {}".format(label))
    print("=" * 60)
    print("  Count:    {}".format(len(w_arr)))
    print("  Mean:     {:.6f}".format(np.mean(w_arr)))
    print("  Std:      {:.6f}".format(np.std(w_arr)))
    print("  Min:      {:.6f}".format(np.min(w_arr)))
    print("  Max:      {:.6f}".format(np.max(w_arr)))
    print("  Median:   {:.6f}".format(np.median(w_arr)))

    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float('inf')]
    hist, _ = np.histogram(w_arr, bins=bins)
    print("\n  Distribution:")
    for j in range(len(hist)):
        lo, hi = bins[j], bins[j + 1]
        bar = '#' * hist[j]
        print("    [{:.1f}, {:.1f}): {:4d}  {}".format(lo, hi, hist[j], bar))
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Helper: tokenize a list of texts → DataLoader
# ---------------------------------------------------------------------------

def tokenize_texts(texts, tokenizer, max_seq_length, batch_size):
    dataset = [tokenizer(t, padding="max_length", max_length=max_seq_length, truncation=True) for t in texts]
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="tummitum/codebert-deprecated", type=str)
    parser.add_argument("--data_path", default="./data/codellama/D_forget.json", type=str)
    parser.add_argument("--dep_data", default="./data/codellama/D_test_U_dep.json", type=str)
    parser.add_argument("--nondep_data", default="./data/codellama/D_test_U_nondep.json", type=str)
    parser.add_argument("--ocsvm_repo", default="tummitum/ocsvm", type=str)
    parser.add_argument("--num_test_samples", default=100, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_file", default="./weight_results.json", type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)

    # --- 1. Load CodeBERT + OCSVM checkpoint ---
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    model = CodeBERTForSelector(args.model_name_or_path, device=device)
    model.to(device)

    ckpt_dir = snapshot_download(args.ocsvm_repo)
    stats = torch.load(os.path.join(ckpt_dir, "stats.pt"), map_location="cpu", weights_only=False)
    mean_list = [m.to(device) for m in stats["mean_list"]]
    precision_list = [p.to(device) for p in stats["precision_list"]]
    fea_list = stats["fea_list"]

    with open(os.path.join(ckpt_dir, "ocsvm.pkl"), "rb") as f:
        c_lr = pickle.load(f)

    # --- 2. D_forget → Mah scores → OCSVM scores → fit Gaussian ---
    dataset, _ = load_dforget(args.data_path, tokenizer, max_seq_length=args.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    print("Loaded {} records from D_forget".format(len(dataset)))

    mah_scores = model.get_unsup_Mah_score(dataloader, mean_list, precision_list, fea_list)[:, 1:]
    d_scores = c_lr.score_samples(mah_scores)
    np.save(args.output_file.replace(".json", "_d_forget_scores.npy"), d_scores)
    mean, std, x0 = weighting_func(d_scores)
    print("Gaussian fit: mean={:.4f}, std={:.4f}, x0={:.4f}".format(mean, std, x0))

    # --- 3. Test on D_test_U_dep (deprecated) ---
    with open(args.dep_data, encoding="utf-8") as f:
        d_dep = json.load(f)
    dep_samples = random.sample(d_dep, min(args.num_test_samples, len(d_dep)))
    dep_texts = [s["probing input"] for s in dep_samples]
    dep_loader = tokenize_texts(dep_texts, tokenizer, args.max_seq_length, args.batch_size)
    dep_mah = model.get_unsup_Mah_score(dep_loader, mean_list, precision_list, fea_list)[:, 1:]
    dep_ocsvm = c_lr.score_samples(dep_mah)
    dep_weights = [obtain_weights(s, mean, std, x0) for s in dep_ocsvm]
    print_weight_stats(dep_weights, "D_test_U_dep (deprecated)")

    # --- 4. Test on D_test_U_nondep (non-deprecated) ---
    with open(args.nondep_data, encoding="utf-8") as f:
        d_nondep = json.load(f)
    nondep_samples = random.sample(d_nondep, min(args.num_test_samples, len(d_nondep)))
    nondep_texts = [s["probing input"] for s in nondep_samples]
    nondep_loader = tokenize_texts(nondep_texts, tokenizer, args.max_seq_length, args.batch_size)
    nondep_mah = model.get_unsup_Mah_score(nondep_loader, mean_list, precision_list, fea_list)[:, 1:]
    nondep_ocsvm = c_lr.score_samples(nondep_mah)
    nondep_weights = [obtain_weights(s, mean, std, x0) for s in nondep_ocsvm]
    print_weight_stats(nondep_weights, "D_test_U_nondep (non-deprecated)")

    # --- 5. Summary ---
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print("  D_test_U_dep:    mean={:.6f}, min={:.6f}, max={:.6f}".format(
        np.mean(dep_weights), np.min(dep_weights), np.max(dep_weights)))
    print("  D_test_U_nondep: mean={:.6f}, min={:.6f}, max={:.6f}".format(
        np.mean(nondep_weights), np.min(nondep_weights), np.max(nondep_weights)))
    print("=" * 60)

    # --- 6. Save results ---
    results = {
        "seed": args.seed,
        "num_test_samples": args.num_test_samples,
        "gaussian_fit": {"mean": float(mean), "std": float(std), "x0": float(x0)},
        "D_test_U_dep": {
            "count": len(dep_weights),
            "mean": float(np.mean(dep_weights)),
            "std": float(np.std(dep_weights)),
            "min": float(np.min(dep_weights)),
            "max": float(np.max(dep_weights)),
            "median": float(np.median(dep_weights)),
        },
        "D_test_U_nondep": {
            "count": len(nondep_weights),
            "mean": float(np.mean(nondep_weights)),
            "std": float(np.std(nondep_weights)),
            "min": float(np.min(nondep_weights)),
            "max": float(np.max(nondep_weights)),
            "median": float(np.median(nondep_weights)),
        },
    }
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to {}".format(args.output_file))


if __name__ == "__main__":
    main()

