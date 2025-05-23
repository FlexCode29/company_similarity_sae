#!/usr/bin/env python3
# strong_features_parallel.py
"""
GPU-parallel “strong latent” miner for clusters.

Examples
────────

# 1 ▸ replay previous run
torchrun --standalone --nproc_per_node 8 \
    strong_features_parallel.py \
    --clusters-pkl year_cluster_dfC-CD.pkl \
    --scores-folder ../fuzzing_scores \
    --reference-pkl ../data/PCA_strong_features_clusters_random_subset.pkl \
    --out strong_features_clusters_random_subset.pkl

# 2 ▸ fresh random subset of 200 clusters
torchrun --standalone --nproc_per_node 8 \
    strong_features_parallel.py \
    --clusters-pkl year_cluster_dfC-CD.pkl \
    --scores-folder ../fuzzing_scores \
    --sample-size 200 \
    --out strong_features_new_random_200.pkl
"""

# ───────────────────────── imports ─────────────────────────
import os, random, argparse, pickle, warnings, re
import numpy as np, pandas as pd
import torch, torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import joblib, datasets, pickle  # noqa: F401  (pickle re-import for type checkers)

torch.backends.cuda.matmul.allow_tf32 = True  # MI250-X likes TF32

# ───────────────────────── helpers ─────────────────────────
_latent_re = re.compile(r"latent(\d+)", re.I)

def extract_latent(fn: str) -> int | None:
    m = _latent_re.search(fn)
    return int(m.group(1)) if m else None

def load_scored_latents(folder: str, n: int = 1000, seed: int = 42) -> np.ndarray:
    """Pick `n` unique latent indices that have score files in `folder`."""
    lat = {extract_latent(f) for f in os.listdir(folder) if f.endswith(".txt")}
    lat.discard(None)
    rng = np.random.default_rng(seed)
    return rng.choice(sorted(lat), size=min(n, len(lat)), replace=False).astype(int)

def cosine_pairs(mat: torch.Tensor, i1, i2):
    n = torch.linalg.norm(mat, dim=1)
    return (mat[i1] * mat[i2]).sum(-1) / (n[i1] * n[i2])

# ───────────────────────── CLI ─────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--clusters-pkl", required=True,
              help="pickle with yearly cluster dicts (e.g. year_cluster_dfC-CD.pkl)")
grp = P.add_mutually_exclusive_group()
grp.add_argument("--reference-pkl",
                 help="previous output pickle – process only those clusters")
grp.add_argument("--sample-size", type=int,
                 help="fresh random subset of this many clusters")
P.add_argument("--pairs-ds",  default="v1ctor10/cos_sim_4000pca_exp")
P.add_argument("--feature-ds",default="marco-molinari/company_reports_with_features")
P.add_argument("--cov-ds",default=("Mateusz1017/annual_reports_tokenized_llama3_logged_returns_"
                                   "no_null_returns_and_incomplete_descriptions_24k"))
P.add_argument("--pca-model",  default="../data/global_pca_model.pkl")
P.add_argument("--scores-folder", default="./fuz_scores")
P.add_argument("--out", default="strong_features_clusters_random_subset.pkl")
args = P.parse_args()

# ───── select clusters ─────
with open(args.clusters_pkl, "rb") as f:
    clusters_df = pickle.load(f)

all_clusters = [c for row in clusters_df["clusters"]
                  for c in row.values() if len(c) > 1]

if args.reference_pkl:
    with open(args.reference_pkl, "rb") as f:
        ref = pickle.load(f)
    ref_set = {frozenset(map(int, k.split(","))) for k in ref}
    all_clusters = [c for c in all_clusters if frozenset(c) in ref_set]
    print(f"[info] {len(all_clusters)} clusters taken from reference pickle")
elif args.sample_size:
    random.seed(42); random.shuffle(all_clusters)
    all_clusters = all_clusters[:args.sample_size]
    print(f"[info] random sample of {len(all_clusters)} clusters")
else:
    print(f"[info] processing every cluster in file ({len(all_clusters)})")

# ───────────────────────── DDP ─────────────────────────
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local_rank  = int(os.environ.get("LOCAL_RANK", 0))
device      = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)
warnings.filterwarnings("ignore")

if rank == 0:
    for _ in tqdm(range(60), desc="⏳ init", total=60, position=0):
        torch.cuda._sleep(int(1e6))
dist.barrier()

# ───────────────────────── load data & PCA ─────────────────────────
df_f = datasets.load_dataset(args.feature_ds)["train"].to_pandas()
df_c = datasets.load_dataset(args.cov_ds)["train"].to_pandas()
df   = (pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
          .dropna(subset=["sic_code", "features"]))
df["features"] = df["features"].apply(lambda x: np.asarray(x[0], dtype=np.float32))

scaler = StandardScaler().fit(np.vstack(df["features"].values))
pca    = joblib.load(args.pca_model)       # 4000 PCs

mean_t  = torch.tensor(scaler.mean_,  dtype=torch.float32, device=device)
scale_t = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
W_t     = torch.tensor(pca.components_.T, dtype=torch.float32, device=device)  # (D,P)

pairs_full = datasets.load_dataset(args.pairs_ds)["train"].to_pandas()
cand_idx   = load_scored_latents(args.scores_folder, 1000)
vec_dim    = df["features"].iat[0].shape[0]

# ───────────────────────── main loop ─────────────────────────
my_clusters = [c for i, c in enumerate(all_clusters) if i % world == rank]
pbar = tqdm(my_clusters, desc=f"rank{rank}", position=rank, leave=False)

local_out = {}
for firms in pbar:
    firms = set(firms)
    pairs = pairs_full[pairs_full["Company1"].isin(firms) &
                       pairs_full["Company2"].isin(firms)]
    if pairs.empty:
        continue
    pairs = pairs.sample(n=min(200, len(pairs)), random_state=42)

    mask_rows = (df["__index_level_0__"].astype(str) + " " + df["year"]).isin(
        pd.concat([pairs["Company1"].astype(str) + " " + pairs["year"],
                   pairs["Company2"].astype(str) + " " + pairs["year"]]))
    raw_np   = np.vstack(df.loc[mask_rows, "features"].values).astype(np.float32)
    comp_idx = dict(zip(df.loc[mask_rows, "__index_level_0__"], range(len(raw_np))))

    X_scaled = (torch.tensor(raw_np, device=device) - mean_t) / scale_t       # (N,D)
    Z        = X_scaled @ W_t                                                # (N,P)

    idx1 = torch.tensor(pairs["Company1"].map(comp_idx).values,
                        device=device, dtype=torch.long)
    idx2 = torch.tensor(pairs["Company2"].map(comp_idx).values,
                        device=device, dtype=torch.long)
    true_cs = cosine_pairs(Z, idx1, idx2).detach()

    # ── MAE for each candidate latent ──
    idx_to_mae = {}
    for j in cand_idx:
        if j >= vec_dim:
            continue
        raw_j  = torch.tensor(raw_np[:, j], device=device)
        delta  = (-(raw_j / scale_t[j])).unsqueeze(1) * W_t[j]               # (N,P)
        mae    = F.l1_loss(cosine_pairs(Z + delta, idx1, idx2), true_cs,
                           reduction="mean").item()
        idx_to_mae[j] = mae

    res = pd.Series(idx_to_mae, name="mae").to_frame()

    # ── Robust adaptive elbow search ──
    quantiles = (0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50)
    strong_idx = weak_idx = None

    def mae_with_set(idx_list):
        if len(idx_list) == 0:
            return 0.0
        delta = torch.zeros_like(Z)
        for j in idx_list:
            raw_j = torch.tensor(raw_np[:, j], device=device)
            delta += (-(raw_j / scale_t[j])).unsqueeze(1) * W_t[j]
        return F.l1_loss(cosine_pairs(Z + delta, idx1, idx2), true_cs,
                         reduction="mean").item()

    for q in quantiles:
        thr        = res["mae"].quantile(q)
        strong_idx = res[res["mae"] >  thr].index.to_numpy(int)
        weak_idx   = res[res["mae"] <= thr].index.to_numpy(int)
        if len(strong_idx) == 0 or len(weak_idx) == 0:
            continue
        mae_strong = mae_with_set(strong_idx)
        mae_weak   = mae_with_set(weak_idx)

        if mae_weak == 0:
            if mae_strong == 0:   # both zero – meaningless, test next q
                continue
            break                 # weak 0 & strong >0  → good elbow
        if mae_strong / mae_weak > 1:
            break                 # standard elbow condition
    else:
        # fallback: fixed 90-th percentile
        thr        = res["mae"].quantile(0.90)
        strong_idx = res[res["mae"] > thr].index.to_numpy(int)
        weak_idx   = res[res["mae"] <= thr].index.to_numpy(int)

    key = ",".join(sorted(map(str, firms)))
    local_out[key] = {
        "strong_features": strong_idx,
        "weak_features":   weak_idx,
        "mae_by_feature":  res
    }

# ───────────────────────── gather & save ─────────────────────────
gather = [None] * world
dist.gather_object(local_out, gather if rank == 0 else None, dst=0)

if rank == 0:
    merged = {}
    for d in gather:
        merged.update(d)
    with open(args.out, "wb") as f:
        pickle.dump(merged, f)
    print(f"\n✅ wrote '{args.out}' with {len(merged)} clusters.")

dist.barrier()
dist.destroy_process_group()
