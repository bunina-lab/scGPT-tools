"""
Fine-tune scGPT on a reference dataset (batch integration / MLM objective).

Fixes vs previous version:
  1. GeneVocab has no .get() — use vocab.get_stoi() to obtain a plain dict first.
  2. Use scgpt.preprocess.Preprocessor instead of manual binning.
  3. Use scgpt.tokenizer.tokenize_and_pad_batch + random_mask_value instead of
     a hand-rolled SeqDataset.

Usage:
    PYTHONNOUSERSITE=1 python run_finetune_integration.py \
        --data_path /path/to/reference.h5ad \
        --model_dir /path/to/scGPT_checkpoint \
        --save_dir  /path/to/output \
        --batch_key batch \
        --n_hvg 3000 \
        --epochs 15 \
        --batch_size 64
"""

import argparse
import json
import os
import shutil
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
    p = argparse.ArgumentParser(description="Fine-tune scGPT – batch integration")

    # I/O
    p.add_argument("--data_path",  required=True, help="Path to reference .h5ad")
    p.add_argument("--model_dir",  required=True,
                   help="Pretrained checkpoint dir (best_model.pt, args.json, vocab.json)")
    p.add_argument("--save_dir",   required=True, help="Output directory")

    # Data
    p.add_argument("--batch_key",     default="batch",
                   help="adata.obs column for batch/sample identity")
    p.add_argument("--n_hvg",         type=int, default=3000,
                   help="Number of highly variable genes (0 = use all)")
    p.add_argument("--max_seq_len",   type=int, default=1201,
                   help="Max gene tokens per cell (n_hvg + 1 for <cls>)")
    p.add_argument("--n_bins",        type=int, default=51,
                   help="Number of expression bins (must match pretrained model)")
    p.add_argument("--mask_ratio",    type=float, default=0.15)
    p.add_argument("--mask_value",    type=int, default=-1,
                   help="Integer marking masked positions")
    p.add_argument("--pad_value",     type=int, default=-2,
                   help="Integer used for expression padding")
    p.add_argument("--include_zero_gene", action="store_true", default=False,
                   help="Include zero-expressed genes in the token sequence")
    p.add_argument("--celltype_column", type=str, help="celltype column name in the adata.obs", required=False, default=None)

    # Training
    p.add_argument("--epochs",        type=int,   default=15)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-5)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--task",  type=str, choices=['grn', 'cell_annotation', 'perturbation', 'batch', 'general'], default='grn')
    p.add_argument("--batch_correction", dest='dsbn', action='store_true')
    p.add_argument("--scheduler",     choices=["cosine", "linear", "none"],
                   default="cosine")

    # Model
    p.add_argument("--use_flash_attn", action="store_true", default=False)
    #p.add_argument("--dsbn",           action="store_true",  default=True,
    #               help="Domain-specific batch normalisation")
    #p.add_argument("--no_dsbn",        dest="dsbn", action="store_false")

    # Misc
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--log_interval",  type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--wandb",         action="store_true", default=False)
    p.add_argument("--wandb_project", default="scGPT_finetune")
    p.add_argument("--wandb_run",     default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_config(model_dir: Path) -> dict:
    cfg = model_dir / "args.json"
    if not cfg.exists():
        sys.exit(f"[error] args.json not found in {model_dir}")
    with open(cfg) as f:
        return json.load(f)


def preprocess_adata(adata, batch_key: str, n_hvg: int, n_bins: int, Preprocessor):
    """
    Use the official scGPT Preprocessor:
      normalize → log1p → HVG selection → binning
    Result stored in adata.layers["X_binned"].
    """
    import scanpy as sc

    print(f"[data] Raw shape: {adata.shape}")
    #sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"[data] After QC:  {adata.shape}")

    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
        normalize_total=1e4,
        result_normed_key="X_normed",
        log1p=True,
        result_log1p_key="X_log1p",
        subset_hvg=n_hvg if n_hvg > 0 else False,
        hvg_flavor="seurat_v3",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata, batch_key=batch_key if batch_key in adata.obs else None)
    print(f"[data] After HVG: {adata.shape}")
    return adata


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SeqDataset(Dataset):
    """Wraps pre-tokenised tensors; __getitem__ is O(1)."""

    def __init__(self, data: dict, batch_ids: np.ndarray, celltype_ids: np.ndarray = None, celltype_labels: np.ndarray = None):
        self.gene_ids     = data["genes"]
        self.expr_values  = data["values"]
        self.target_expr  = data["target_values"]
        self.batch_ids    = torch.from_numpy(batch_ids).long()
        
        # Add celltype_ids if provided
        if celltype_ids is not None:
            self.celltype_ids = torch.from_numpy(celltype_ids).long()
            self.celltype_labels = torch.from_numpy(celltype_labels).long()
        else:
            self.celltype_ids = None
            self.celltype_labels = None

    def __len__(self):
        return self.gene_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "gene_ids":     self.gene_ids[idx],
            "expr":         self.expr_values[idx],
            "target_expr":  self.target_expr[idx],
            "batch_id":     self.batch_ids[idx],
        }
        # Safely add celltype_id to the batch dictionary
        if self.celltype_ids is not None:
            item["celltype_ids"] = self.celltype_ids[idx]
            item["celltype_labels"] = self.celltype_labels[idx]
        return item


def build_dataloader(adata, vocab, gene2idx: dict, batch_ids: np.ndarray,
                     args, celltype_ids: np.ndarray = None, celltype_labels:np.ndarray = None, shuffle: bool = True):
    from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
    import scipy.sparse as sp

    pad_id = gene2idx[args.pad_token]

    # Map dataset gene names → vocab integer ids via the plain dict
    gene_ids_arr = np.array(
        [gene2idx.get(g, pad_id) for g in adata.var_names],
        dtype=np.int64,
    )

    X = adata.layers["X_binned"]
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    tokenised = tokenize_and_pad_batch(
        data=X,
        gene_ids=gene_ids_arr,
        max_len=args.max_seq_len,
        vocab=vocab,
        pad_token=args.pad_token,
        pad_value=args.pad_value,
        append_cls=True,
        include_zero_gene=args.include_zero_gene,
    )

    target_values = tokenised["values"].clone()
    masked_values = random_mask_value(
        tokenised["values"],
        mask_ratio=args.mask_ratio,
        mask_value=args.mask_value,
        pad_value=args.pad_value,
    )
    tokenised["values"]        = masked_values
    tokenised["target_values"] = target_values

    # Pass the celltype_ids to the dataset
    ds = SeqDataset(
        data=tokenised,
        batch_ids=batch_ids, 
        celltype_ids=celltype_ids,
        celltype_labels=celltype_labels
        )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def get_scheduler(optimizer, args, n_steps_total: int):
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    if args.scheduler == "none":
        return None
    warmup_steps = int(n_steps_total * args.warmup_ratio)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_steps)
    if args.scheduler == "cosine":
        main = CosineAnnealingLR(optimizer, T_max=n_steps_total - warmup_steps,
                                 eta_min=args.lr * 0.01)
    else:
        main = LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                        total_iters=n_steps_total - warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, main],
                        milestones=[warmup_steps])


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, criterion, criterion_cls, scaler,
                device, args, pad_token_id, step: int, logger=None):
    model.train()
    total_loss = 0.0

   # --- 1. Configure Objectives Based on Task ---
    if args.task == 'grn' or args.task == 'cell_annotation': 
        # Cell-Type Annotation & GRNs
        CLS = True
        MVC = True
        ECS = False
        ADV = False
        CCE = False
        DAB = False
        
    elif args.task == 'batch': 
        # Batch Integration
        CLS = False
        MVC = True
        ECS = True
        ADV = True   # DAB/Adversarial is ON to strip batch signals
        DSBN = True
        CCE = False
        DAB = True

    elif args.task == 'perturbation': 
        # Perturbation Prediction
        CLS = False
        MVC = True
        ECS = False
        ADV = False
        #DSBN = False
        CCE = False
        DAB = True

    elif args.task == 'general': 
        # General Pre-training
        CLS = False
        MVC = True
        ECS = False
        ADV = False
        CCE = False
        DAB = False
    else:
        raise ValueError(f"Unknown task: {args.task}")


    for i, batch in enumerate(loader):
        gene_ids     = batch["gene_ids"].to(device)
        expr         = batch["expr"].to(device)
        target_expr  = batch["target_expr"].to(device)
        key_pad_mask = gene_ids.eq(pad_token_id)   # True where gene_id == <pad>
        batch_id     = batch["batch_id"].to(device)
        
        if CLS:
            celltype_labels = batch["celltype_labels"].to(device)
            celltype_id = batch["celltype_ids"].to(device)

        optimizer.zero_grad()

        output_dict = model(
            src=gene_ids,
            values=expr,
            src_key_padding_mask=key_pad_mask,
            batch_labels=batch_id if args.dsbn else None,
            CLS=CLS, # Cell-type classification
            MVC=MVC, # Masked value prediction
            ECS=ECS, # Elastic Cell Similarity for batch integration
            CCE=CCE,  # Contrastive cell embedding objective ## TODO:not implemented atm,
        )

        mlm_output = output_dict["mlm_output"]   # (B, seq_len, n_bins)
        loss_mask  = ~key_pad_mask                # valid positions

        
        explicit_zero_prob = args.include_zero_gene  # whether explicit bernoulli for zeros

        ### LOSS calculations
        metrics_to_log = {}
        # Calculate base MLM loss
        loss = criterion(mlm_output, target_expr.float(), loss_mask)
        metrics_to_log = {"train/mse": loss.item()}

        ###print(output_dict) ## DEBUG purposes

        masked_positions = expr.eq(args.mask_value)

        if explicit_zero_prob and "mlm_zero_probs" in output_dict.keys():
            from scgpt.loss import criterion_neg_log_bernoulli
            loss_zero_log_prob = criterion_neg_log_bernoulli(
                output_dict["mlm_zero_probs"], target_expr, masked_positions
            )
            loss = loss + loss_zero_log_prob
            metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
        if CLS:
            loss_cls = criterion_cls(output_dict["cls_output"], celltype_labels)
            loss = loss + loss_cls*2 ## Added weight for classification
            metrics_to_log.update({"train/cls": loss_cls.item()})

            error_rate = 1 - (
                (output_dict["cls_output"].argmax(1) == celltype_labels)
                .sum()
                .item()
            ) / celltype_labels.size(0)
        if CCE and "loss_cce" in output_dict.keys():
            loss_cce = 10 * output_dict["loss_cce"]
            loss = loss + loss_cce
            metrics_to_log.update({"train/cce": loss_cce.item()})
        if MVC and "mvc_output" in output_dict.keys():
            loss_mvc = criterion(
                output_dict["mvc_output"], target_expr, masked_positions
            )
            loss = loss + loss_mvc
            metrics_to_log.update({"train/mvc": loss_mvc.item()})
        if MVC and explicit_zero_prob and "mvc_output" in output_dict.keys() and "mlm_zero_probs" in output_dict.keys():
            from scgpt.loss import criterion_neg_log_bernoulli
            loss_mvc_zero_log_prob = criterion_neg_log_bernoulli(
                output_dict["mvc_zero_probs"], target_expr, masked_positions
            )
            loss = loss + loss_mvc_zero_log_prob
            metrics_to_log.update({"train/mvc_nzlp": loss_mvc_zero_log_prob.item()})
        if ECS and "loss_ecs" in output_dict.keys():
            loss_ecs = 10 * output_dict["loss_ecs"]
            loss = loss + loss_ecs
            metrics_to_log.update({"train/ecs": loss_ecs.item()})
        if DAB and "dab_output" in output_dict.keys():
            # try weighting and separate optimizer
            loss_dab = criterion_dab(output_dict["dab_output"], batch_id)
            dab_weight = 1
            loss = loss + dab_weight * loss_dab
            metrics_to_log.update({"train/dab": loss_dab.item()})

        
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()


        ##loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        #optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        step += 1

        if (i + 1) % args.log_interval == 0:
            avg    = total_loss / (i + 1)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  step {i+1}/{len(loader)}  loss={avg:.4f}  lr={lr_now:.2e}")
            if logger:
                logger.log({"train/loss": avg, "train/lr": lr_now}, step=step)

    return total_loss / len(loader), step


@torch.no_grad()
def eval_epoch(model, loader, criterion, criterion_cls, device, args, pad_token_id):
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_error = 0.0
    total_num_cells = 0
    total_num = 0

    # --- 1. Task Logic for Evaluation ---
    if args.task == 'grn' or args.task == 'cell_annotation': 
        EVAL_CLS = True
        EVAL_MSE = False
    elif args.task == 'batch' or args.task == 'general': 
        EVAL_CLS = False
        EVAL_MSE = True
    
    elif args.task == "perturbation":
        raise ValueError(f"Task perturbation is not included atm")
    
    else:
        raise ValueError(f"Unknown task: {args.task}")


    for batch in loader:
        gene_ids     = batch["gene_ids"].to(device)
        expr         = batch["expr"].to(device)
        target_expr  = batch["target_expr"].to(device)
        batch_id     = batch["batch_id"].to(device)
        
        # Safely extract celltype_id if it exists
        celltype_labels = None
        celltype_id = None
        if EVAL_CLS:
            celltype_labels = batch["celltype_labels"].to(device)
            celltype_id = batch["celltype_ids"].to(device)

        key_pad_mask = gene_ids.eq(pad_token_id)
        batch_size = len(gene_ids)

        with torch.cuda.amp.autocast(enabled=True):
            # Notice: Auxiliary tasks (CCE, MVC, ECS) are OFF. 
            # We only evaluate the primary outputs (mlm_output or cls_output).
            output_dict = model(
                src=gene_ids,
                values=expr,
                src_key_padding_mask=key_pad_mask,
                batch_labels=batch_id if args.dsbn else None,
                CLS=EVAL_CLS, 
                MVC=False, 
                ECS=False, 
            )

            loss = 0.0

            # --- Task 1: Evaluate Classification (GRN / Annotation) ---
            if EVAL_CLS and "cls_output" in output_dict and celltype_id is not None:
                cls_logits = output_dict["cls_output"]
                loss_cls = criterion_cls(cls_logits, celltype_id)
                
                total_loss += loss_cls.item() * batch_size
                total_num += batch_size
            
                # Calculate Accuracy/Error Rate
                predictions = cls_logits.argmax(dim=1)
                correct = (predictions == celltype_id).sum().item()
                total_error += (batch_size - correct)

            # --- Task 2: Evaluate Reconstruction (Batch / Perturbation) ---
            if EVAL_MSE and "mlm_output" in output_dict:
                mlm_output = output_dict["mlm_output"]
                loss_mask  = ~key_pad_mask
                loss_mse = criterion(mlm_output, target_expr.float(), loss_mask)
                
                total_mse += loss_mse.item() * batch_size 
                total_loss = total_mse
                total_num += batch_size
    
    if EVAL_CLS:
        print(f"Accuracy: {1.0- (total_error/total_num)}\n")
    if EVAL_MSE:
        print(f"MSE: {(total_mse/total_num)}\n")

    return total_loss / total_num


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    args.pad_token     = "<pad>"
    args.cls_token     = "<cls>"
    args.special_tokens = [args.pad_token, args.cls_token, "<eoc>"]


    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "run_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run or f"finetune_{Path(args.data_path).stem}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
        logger = wandb

    # --- scGPT imports ---
    try:
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from scgpt.preprocess import Preprocessor
        from scgpt.utils import load_pretrained
    except ImportError as e:
        sys.exit(f"[error] Cannot import scgpt: {e}\n"
                 "Activate the right conda env and set PYTHONNOUSERSITE=1.")

    # --- Vocab ---
    model_dir  = Path(args.model_dir)
    vocab_path = model_dir / "vocab.json"
    if not vocab_path.exists():
        sys.exit(f"[error] vocab.json not found in {model_dir}")

    vocab = GeneVocab.from_file(vocab_path)
    for s in args.special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # THE FIX: get a plain Python dict from the GeneVocab object,
    # then call .get() on the dict — not on the vocab object itself.
    gene2idx: dict = vocab.get_stoi()
    pad_token_id   = gene2idx[args.pad_token]
    print(f"[vocab] Size: {len(vocab)}  |  pad_id: {pad_token_id}")

    # --- Pretrained config ---
    pretrain_cfg = load_pretrained_config(model_dir)
    embsize = pretrain_cfg.setdefault("embsize", 512)
    nhead = pretrain_cfg.setdefault("nheads", 8)
    d_hid = pretrain_cfg.setdefault("d_hid", 512)
    nlayers = pretrain_cfg.setdefault("nlayers", 12)
    dropout = pretrain_cfg.setdefault("dropout", 0.0)
    print(f"[model] embsize={embsize}  nhead={nhead}  nlayers={nlayers}")

    # --- Data ---
    import anndata as ad
    print(f"[data] Loading {args.data_path} ...")
    adata = ad.read_h5ad(args.data_path)

    if args.batch_key not in adata.obs.columns:
        sys.exit(f"[error] batch_key='{args.batch_key}' not in adata.obs.\n"
                 f"Available columns: {list(adata.obs.columns)}")

    adata = preprocess_adata(
        adata, args.batch_key, args.n_hvg, args.n_bins, Preprocessor
    )

    batch_cat = adata.obs[args.batch_key].astype("category")
    n_batches = len(batch_cat.cat.categories)
    batch_ids = batch_cat.cat.codes.values.astype(np.int64)
    print(f"[data] {n_batches} batches: {list(batch_cat.cat.categories)}")

    n_match = sum(1 for g in adata.var_names if g in gene2idx)
    print(f"[vocab] {n_match}/{adata.n_vars} dataset genes matched in vocab")


    num_classes = None
    celltype_ids = None
    celltype_labels = None
   # --- 1. Configure Objectives Based on Task ---
    if args.task == 'grn' or args.task == 'cell_annotation': 
        # Cell-Type Annotation & GRNs
        CLS = True
        MVC = True
        ECS = False
        ADV = False
        
        if args.celltype_column not in adata.obs:
            raise ValueError(f"{args.task} requires --celltype_column")
        
        # Convert string cell types to integer categorical codes
        celltype_cat = adata.obs[args.celltype_column].astype("category")
        num_classes = len(celltype_cat.cat.categories) # BUG FIX: len() instead of int()
        celltype_ids = celltype_cat.cat.codes.values.astype(np.int64)
        celltype_labels = celltype_cat.cat.codes.values
        print(f"[data] Found {num_classes} distinct cell types.")

        
    elif args.task == 'batch': 
        # Batch Integration
        CLS = False
        MVC = True
        ECS = True
        ADV = True   # DAB/Adversarial is ON to strip batch signals
        DSBN = True

    elif args.task == 'perturbation': 
        # Perturbation Prediction
        CLS = False
        MVC = True
        ECS = False
        ADV = False
        #DSBN = False

    elif args.task == 'general': 
        # General Pre-training
        CLS = False
        MVC = True
        ECS = False
        ADV = False
    else:
        raise ValueError(f"Unknown task: {args.task}")


    # Train / val split
    if (args.task in ['grn', 'cell_annotation']):
        from sklearn.model_selection import train_test_split
        # Ensure celltype_ids is available for stratification
        labels_for_split = celltype_ids
        train_idx, val_idx = train_test_split(
            np.arange(adata.n_obs), 
            test_size=0.1, 
            random_state=args.seed,
            stratify=labels_for_split
        )
    else:
        rng = np.random.default_rng(args.seed)
        idx       = np.random.permutation(adata.n_obs)
        split     = int(0.9 * adata.n_obs)
        train_idx = idx[:split]
        val_idx   = idx[split:]

    print(f"[data] Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    
    print("[data] Tokenising training set ...")
    train_loader = build_dataloader(
        adata[train_idx], vocab, gene2idx, batch_ids[train_idx], args, 
        celltype_ids=celltype_ids[train_idx] if celltype_ids is not None else None,
        celltype_labels = celltype_labels[train_idx] if celltype_labels is not None else None, 
        shuffle=True
    )
    print("[data] Tokenising validation set ...")
    val_loader = build_dataloader(
        adata[val_idx], vocab, gene2idx, batch_ids[val_idx], args, 
        celltype_ids=celltype_ids[val_idx] if celltype_ids is not None else None,
        celltype_labels = celltype_labels[train_idx] if celltype_labels is not None else None,
        shuffle=False
    )

    DSBN = args.dsbn

    # --- 2. Build model ---
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        nlayers_cls=3,
        # If classifying, pass the actual number of cell types; otherwise default to 1
        n_cls=num_classes if CLS else 1, 
        vocab=vocab,
        dropout=dropout,
        pad_token=args.pad_token,
        pad_value=args.pad_value,
        
        # --- DYNAMICALLY MAPPED PARAMETERS ---
        do_mvc=MVC,                   # Now controlled by the logic block above
        do_dab=ADV,                   # Maps to ADV/DAB for batch integration
        use_batch_labels=DSBN,   # Domain-specific batch norm
        domain_spec_batchnorm=DSBN, 
        # -------------------------------------
        
        num_batch_labels=n_batches,
        n_input_bins=args.n_bins,
        explicit_zero_prob=False,
        use_fast_transformer=args.use_flash_attn,
        pre_norm=False,
    )

    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        candidates = list(model_dir.glob("*.pt"))
        if not candidates:
            sys.exit(f"[error] No .pt checkpoint in {model_dir}")
        ckpt_path = candidates[0]
        print(f"[warn] best_model.pt not found; using {ckpt_path.name}")

    load_pretrained(model, torch.load(ckpt_path, map_location="cpu"), verbose=True)
    model = model.to(device)
    print(f"[model] Pretrained weights loaded from {ckpt_path}")

    # --- Optimiser & scheduler ---
    optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    n_steps_total = args.epochs * len(train_loader)
    scheduler     = get_scheduler(optimizer, args, n_steps_total)

    # Setup your criterion
    from scgpt.loss import masked_mse_loss
    from scgpt.loss import criterion_neg_log_bernoulli

    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss() # Define the classification loss


    ### Scaler 
    AMP = True
    scaler = torch.cuda.amp.GradScaler(enabled=AMP) # Automatic Mixed Precision

    # --- Training loop ---
    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}\nEpoch {epoch}/{args.epochs}")
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, criterion_cls,scaler,
            device, args, pad_token_id, global_step, logger
        )
        print(f"  → Train loss: {train_loss:.4f}")

        if epoch % args.eval_interval == 0:
            val_loss = eval_epoch(model, val_loader, criterion,criterion_cls, device, args, pad_token_id)
            print(f"  → Val   loss: {val_loss:.4f}")
            if logger:
                logger.log({"val/loss": val_loss}, step=global_step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_dir / "best_model.pt")
                print(f"  ✓ New best saved → {save_dir / 'best_model.pt'}")

        torch.save(model.state_dict(), save_dir / "last_model.pt")

    # Save config & vocab for downstream use
    final_cfg = {**pretrain_cfg,
                 "num_batch_labels": n_batches,
                 "use_batch_labels": args.dsbn,
                 "n_input_bins": args.n_bins}
    with open(save_dir / "args.json", "w") as f:
        json.dump(final_cfg, f, indent=2)
    shutil.copy(vocab_path, save_dir / "vocab.json")

    adata.write_h5ad(os.path.join(save_dir, "adata_scgpt.h5ad"))

    print(f"\n[done] Best val loss: {best_val_loss:.4f}")
    print(f"       Saved to:      {save_dir}")
    if logger:
        logger.finish()


if __name__ == "__main__":
    main()