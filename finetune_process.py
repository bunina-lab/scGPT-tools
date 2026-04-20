"""
Fine-tune scGPT on a reference dataset (batch integration objective).

Usage:
    PYTHONNOUSERSITE=1 python run_finetune_integration.py \
        --data_path /path/to/reference.h5ad \
        --model_dir /path/to/scGPT_checkpoint \
        --save_dir /path/to/save/finetuned \
        --batch_key batch \
        --n_hvg 3000 \
        --epochs 15 \
        --batch_size 64

The script expects an AnnData (.h5ad) file. Gene names must be in adata.var_names
(or adata.var["gene_name"]) and a batch column must exist in adata.obs.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune scGPT for batch integration")

    # --- I/O ---
    p.add_argument("--data_path", required=True,
                   help="Path to reference .h5ad file")
    p.add_argument("--model_dir", required=True,
                   help="Directory with pretrained scGPT checkpoint "
                        "(best_model.pt + args.json + vocab.json)")
    p.add_argument("--save_dir", required=True,
                   help="Output directory for fine-tuned model")

    # --- Data ---
    p.add_argument("--batch_key", default="batch",
                   help="obs column that encodes batch/sample identity")
    p.add_argument("--celltype_key", default=None,
                   help="obs column for cell type labels (optional, for logging)")
    p.add_argument("--n_hvg", type=int, default=3000,
                   help="Number of highly variable genes to select (0 = use all)")
    p.add_argument("--max_seq_len", type=int, default=1201,
                   help="Maximum number of gene tokens per cell (n_hvg + 1)")
    p.add_argument("--n_bins", type=int, default=51,
                   help="Number of expression value bins")

    # --- Training ---
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--mask_ratio", type=float, default=0.15,
                   help="Fraction of gene tokens to mask during training")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="Fraction of total steps used for LR warm-up")
    p.add_argument("--scheduler", choices=["cosine", "linear", "none"],
                   default="cosine")

    # --- Model ---
    p.add_argument("--use_flash_attn", action="store_true", default=False)
    p.add_argument("--dsbn", action="store_true", default=True,
                   help="Use domain-specific batch normalisation (recommended)")
    p.add_argument("--no_dsbn", dest="dsbn", action="store_false")

    # --- Misc ---
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=100,
                   help="Print training loss every N steps")
    p.add_argument("--eval_interval", type=int, default=1,
                   help="Run validation every N epochs")
    p.add_argument("--wandb", action="store_true", default=False,
                   help="Log to Weights & Biases")
    p.add_argument("--wandb_project", default="scGPT_finetune")
    p.add_argument("--wandb_run", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "args.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No args.json found in {model_dir}")
    with open(cfg_path) as f:
        return json.load(f)


def preprocess_adata(adata, batch_key: str, n_hvg: int, n_bins: int):
    """Minimal pre-processing: HVG selection + normalisation + binning."""

    print(f"[data] Input shape: {adata.shape}")
    import scanpy as sc

    # Basic QC filters (adjust to taste)
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)

    print(f"[data] After QC: {adata.shape}")

    from scgpt.preprocess import Preprocessor
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=3,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=False,  # 3. whether to normalize the raw data and to what sum
        #result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=False,  # 4. whether to log1p the normalized data
        #result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        #hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key=batch_key)

    return adata


class SeqDataset(Dataset):
    """Dataset that tokenises cells on-the-fly."""

    def __init__(self, adata, vocab, batch_ids, max_seq_len: int, mask_ratio: float,
                 pad_token: str = "<pad>", mask_token: str = "<mask>"):
        import scipy.sparse as sp

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.pad_id = vocab[pad_token]
        self.mask_id = vocab[mask_token]
        self.batch_ids = batch_ids

        # Gene ids for genes present in vocab
        gene_names = list(adata.var_names)
        gene_ids = np.array(
            [vocab.get(g, vocab["<pad>"]) for g in gene_names], dtype=np.int64
        )
        self.gene_ids = gene_ids

        # Expression matrix (dense)
        X = adata.layers["X_binned"]
        if sp.issparse(X):
            X = X.toarray()
        self.X = X.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        expr = self.X[idx]  # (n_genes,)
        gene_ids = self.gene_ids  # (n_genes,)

        # Keep expressed genes only (bin > 0)
        nonzero_mask = expr > 0
        expr_nz = expr[nonzero_mask]
        gids_nz = gene_ids[nonzero_mask]

        # Truncate / pad to max_seq_len
        n = min(len(expr_nz), self.max_seq_len)
        if len(expr_nz) > self.max_seq_len:
            idx_sel = np.random.choice(len(expr_nz), self.max_seq_len, replace=False)
            expr_nz = expr_nz[idx_sel]
            gids_nz = gids_nz[idx_sel]
            n = self.max_seq_len

        # Pad
        expr_pad = np.full(self.max_seq_len, self.pad_id, dtype=np.int64)
        gids_pad = np.full(self.max_seq_len, self.pad_id, dtype=np.int64)
        expr_pad[:n] = expr_nz
        gids_pad[:n] = gids_nz

        # Key padding mask (True = padding position)
        key_pad_mask = np.ones(self.max_seq_len, dtype=bool)
        key_pad_mask[:n] = False

        # Masked language model target
        target_expr = expr_pad.copy()
        masked_expr = expr_pad.copy()
        n_mask = max(1, int(n * self.mask_ratio))
        mask_pos = np.random.choice(n, n_mask, replace=False)
        masked_expr[mask_pos] = self.mask_id

        return {
            "gene_ids": torch.from_numpy(gids_pad),
            "expr": torch.from_numpy(masked_expr),
            "target_expr": torch.from_numpy(target_expr),
            "key_pad_mask": torch.from_numpy(key_pad_mask),
            "batch_id": torch.tensor(self.batch_ids[idx], dtype=torch.long),
        }


def build_dataloader(adata, vocab, batch_ids, args, shuffle: bool = True):
    ds = SeqDataset(
        adata=adata,
        vocab=vocab,
        batch_ids=batch_ids,
        max_seq_len=args.max_seq_len,
        mask_ratio=args.mask_ratio,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_scheduler(optimizer, args, n_steps_total: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_steps = int(n_steps_total * args.warmup_ratio)

    if args.scheduler == "none":
        return None

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_steps)
    if args.scheduler == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=n_steps_total - warmup_steps,
                                 eta_min=args.lr * 0.01)
    else:  # linear decay
        main = LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                        total_iters=n_steps_total - warmup_steps)

    return SequentialLR(optimizer, schedulers=[warmup, main],
                        milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, criterion, device,
                n_batches: int, args, step: int, logger=None):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(loader):
        gene_ids = batch["gene_ids"].to(device)
        expr = batch["expr"].to(device)
        target_expr = batch["target_expr"].to(device)
        key_pad_mask = batch["key_pad_mask"].to(device)
        batch_id = batch["batch_id"].to(device)

        optimizer.zero_grad()

        output_dict = model(
            src=gene_ids,
            values=expr,
            src_key_padding_mask=key_pad_mask,
            batch_labels=batch_id if args.dsbn else None,
            CLS=False,
            MVC=False,
            ECS=False,
        )

        mlm_output = output_dict["mlm_output"]  # (B, seq_len, n_bins)

        # Only compute loss on non-padding positions
        loss_mask = ~key_pad_mask  # (B, seq_len)
        loss = criterion(
            mlm_output[loss_mask],  # (N_valid, n_bins)
            target_expr[loss_mask],  # (N_valid,)
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        step += 1

        if (i + 1) % args.log_interval == 0:
            avg = total_loss / (i + 1)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  step {i+1}/{n_batches}  loss={avg:.4f}  lr={lr_now:.2e}")
            if logger:
                logger.log({"train/loss": avg, "train/lr": lr_now}, step=step)

    return total_loss / len(loader), step


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, args):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        expr = batch["expr"].to(device)
        target_expr = batch["target_expr"].to(device)
        key_pad_mask = batch["key_pad_mask"].to(device)
        batch_id = batch["batch_id"].to(device)

        output_dict = model(
            src=gene_ids,
            values=expr,
            src_key_padding_mask=key_pad_mask,
            batch_labels=batch_id if args.dsbn else None,
            CLS=False,
            MVC=False,
            ECS=False,
        )

        mlm_output = output_dict["mlm_output"]
        loss_mask = ~key_pad_mask
        loss = criterion(mlm_output[loss_mask], target_expr[loss_mask])
        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(save_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Optional W&B ---
    logger = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run or f"finetune_{Path(args.data_path).stem}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        logger = wandb

    # --- scGPT imports (after PYTHONNOUSERSITE guard) ---
    try:
        import scgpt
        from scgpt.model import TransformerModel
        from scgpt.tokenizer import GeneVocab
        from scgpt.utils import load_pretrained
    except ImportError as e:
        sys.exit(f"[error] Could not import scgpt: {e}\n"
                 "Make sure you activated the correct conda environment and "
                 "set PYTHONNOUSERSITE=1.")

    # --- Load vocab ---
    model_dir = Path(args.model_dir)
    vocab_path = model_dir / "vocab.json"
    if not vocab_path.exists():
        sys.exit(f"[error] vocab.json not found in {model_dir}")
    vocab = GeneVocab.from_file(vocab_path)
    print(f"[vocab] Size: {len(vocab)}")

    # --- Load pretrained model config ---
    pretrain_cfg = load_pretrained_config(model_dir)
    embsize     = pretrain_cfg.get("embsize", 512)
    nhead       = pretrain_cfg.get("nheads", 8)
    d_hid       = pretrain_cfg.get("d_hid", 512)
    nlayers     = pretrain_cfg.get("nlayers", 12)
    dropout     = pretrain_cfg.get("dropout", 0.0)
    print(f"[model] embsize={embsize}, nhead={nhead}, nlayers={nlayers}")

    # --- Load and preprocess data ---
    import anndata as ad
    print(f"[data] Loading {args.data_path} ...")
    adata = ad.read_h5ad(args.data_path)

    if args.batch_key not in adata.obs.columns:
        sys.exit(f"[error] batch_key '{args.batch_key}' not found in adata.obs. "
                 f"Available columns: {list(adata.obs.columns)}")

    adata = preprocess_adata(adata, args.batch_key, args.n_hvg, args.n_bins)

    # Encode batch labels as integers
    batch_categories = adata.obs[args.batch_key].astype("category")
    n_batches = len(batch_categories.cat.categories)
    batch_ids = batch_categories.cat.codes.values.astype(np.int64)
    print(f"[data] {n_batches} batches: {list(batch_categories.cat.categories)}")

    # Train / val split (90/10 by cell)
    n_cells = adata.n_obs
    idx = np.random.permutation(n_cells)
    split = int(0.9 * n_cells)
    train_idx, val_idx = idx[:split], idx[split:]
    adata_train = adata[train_idx]
    adata_val   = adata[val_idx]
    print(f"[data] Train: {len(train_idx)} cells | Val: {len(val_idx)} cells")

    train_loader = build_dataloader(adata_train, vocab, batch_ids[train_idx],
                                    args, shuffle=True)
    val_loader   = build_dataloader(adata_val,   vocab, batch_ids[val_idx],
                                    args, shuffle=False)

    # --- Build model ---
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        nlayers_cls=3,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=vocab["<pad>"],
        pad_value=0,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=args.dsbn,
        num_batch_labels=n_batches,
        domain_spec_batchnorm=args.dsbn,
        n_input_bins=args.n_bins,
        explicit_zero_prob=False,
        use_fast_transformer=args.use_flash_attn,
        pre_norm=False,
    )

    # Load pretrained weights (ignores mismatched keys gracefully)
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        # Try alternative names
        candidates = list(model_dir.glob("*.pt"))
        if not candidates:
            sys.exit(f"[error] No .pt checkpoint found in {model_dir}")
        ckpt_path = candidates[0]
        print(f"[warn] best_model.pt not found; using {ckpt_path.name}")

    load_pretrained(model, torch.load(ckpt_path, map_location="cpu"),
                    verbose=True)
    model = model.to(device)
    print(f"[model] Loaded pretrained weights from {ckpt_path}")

    # --- Optimizer & scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    n_steps_total = args.epochs * len(train_loader)
    scheduler = get_scheduler(optimizer, args, n_steps_total)

    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}")
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, len(train_loader), args, global_step, logger
        )
        print(f"  → Train loss: {train_loss:.4f}")

        if epoch % args.eval_interval == 0:
            val_loss = eval_epoch(model, val_loader, criterion, device, args)
            print(f"  → Val   loss: {val_loss:.4f}")
            if logger:
                logger.log({"val/loss": val_loss}, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_out = save_dir / "best_model.pt"
                torch.save(model.state_dict(), ckpt_out)
                print(f"  ✓ New best model saved → {ckpt_out}")

        # Save latest checkpoint every epoch
        torch.save(model.state_dict(), save_dir / "last_model.pt")

    # Save model args so downstream annotation scripts can reload correctly
    final_args = {**pretrain_cfg,
                  "num_batch_labels": n_batches,
                  "use_batch_labels": args.dsbn,
                  "n_input_bins": args.n_bins}
    with open(save_dir / "args.json", "w") as f:
        json.dump(final_args, f, indent=2)

    # Copy vocab to save_dir for convenience
    import shutil
    shutil.copy(vocab_path, save_dir / "vocab.json")

    print(f"\n[done] Fine-tuning complete. Best val loss: {best_val_loss:.4f}")
    print(f"       Model saved to: {save_dir}")
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()
