import argparse
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
    """Fit a single Gaussian on D_forget OCSVM scores.
    (Original weighting_func_gmm fitted 2 Gaussians from train_scores + test_scores)
    """
    mean, std = norm.fit(d_forget_scores)
    x0 = mean  # median of Gaussian = mean
    return mean, std, x0


def obtain_weights(input_x, mean, std, x0):
    """Compute soft weight for a single OCSVM score.
    Formula identical to original obtain_weights:
      norm.cdf replaces gmm_cdf, rest unchanged.
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
# Main: load checkpoint → fit Gaussian → compute weights
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="tummitum/codebert-deprecated", type=str)
    parser.add_argument("--data_path", default="./data/codellama/D_forget.json", type=str)
    parser.add_argument("--ocsvm_repo", default="tummitum/ocsvm", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", type=int, default=2026)
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
    print("Loaded {} records from {}".format(len(dataset), args.data_path))

    mah_scores = model.get_unsup_Mah_score(dataloader, mean_list, precision_list, fea_list)[:, 1:]
    d_scores = c_lr.score_samples(mah_scores)
    mean, std, x0 = weighting_func(d_scores)
    print("Gaussian fit: mean={:.4f}, std={:.4f}, x0={:.4f}".format(mean, std, x0))

    # --- 3. Compute weights for D_forget ---
    weights = [obtain_weights(s, mean, std, x0) for s in d_scores]
    print("D_forget weights: mean={:.4f}, min={:.4f}, max={:.4f}".format(
        np.mean(weights), np.min(weights), np.max(weights)))


if __name__ == "__main__":
    main()
