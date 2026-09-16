"""In-silico perturbation of the attention GRN input.

The pipeline builds ``total_embs = encoder(gene_ids) + value_encoder(values)``
before extracting attention (see ``lib/scgpt_model.py``). Perturbation is
therefore a pure edit of the tokenized value matrix: no new model, no new
dependency, NO FINE-TUNING. The same frozen, eval-mode weights run on the
perturbed input; only one column differs from an ordinary GRN run.

Levels are expressed in bin space because ``X_binned`` holds scGPT's per-cell
QUANTILE bins, not counts. A bin is an ordinal rank within one cell, so a
fold-change is not expressible here: multiplying a bin is a category error
(bin 20 is not "twice" bin 10), and re-binning from raw counts would move the
per-cell quantile boundaries and shift other genes' bins too, destroying the
single-gene intervention.

We also never remove the gene's token, only overwrite its value, so the
sequence length M stays constant and the attention ranks remain comparable
with an unperturbed run.
"""
import numpy as np
import torch

KINDS = ("bin", "quantile", "max", "sham")


def parse_level(level):
    """Parse a ``--perturb-level`` string.

    Returns one of ``('bin', n)``, ``('quantile', p)``, ``('max', None)`` or
    ``('sham', None)``.

    Note ``q:0`` is the *lowest nonzero* bin, not a knockout; use ``bin:0``
    (or its alias ``ko``) to make a gene read as not expressed.
    """
    if level == "sham":
        return ("sham", None)
    if level == "ko":
        return ("bin", 0.0)
    if level == "max":
        return ("max", None)
    if level.startswith("bin:"):
        return ("bin", float(level[len("bin:"):]))
    if level.startswith("q:"):
        q = float(level[len("q:"):])
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, got {q}")
        return ("quantile", q)
    raise ValueError(
        f"Unrecognised level {level!r}. Use bin:N, q:P, ko, max or sham. "
        f"Fold-changes are not expressible on per-cell quantile bins."
    )


def level_slug(level):
    """Filename-safe form of a level string."""
    return level.replace(":", "")


def gene_token_position(genes, gene_name):
    """Index of ``gene_name`` in the tokenized sequence, including the <cls> offset.

    ``tokenize_and_pad_batch(append_cls=True, include_zero_gene=True)`` preserves
    gene order and prepends <cls>, so the position is the index in ``genes`` + 1.
    """
    hits = np.where(np.asarray(genes) == gene_name)[0]
    if hits.size == 0:
        raise KeyError(f"Gene {gene_name!r} is not in the adata var set")
    return int(hits[0]) + 1


def target_bins(values, kind, param, max_bin):
    """Per-cell target bin value. Returns shape ``(n_cells,)``.

    For ``quantile`` the target is derived from each cell's own nonzero binned
    values, excluding the <cls> token at position 0, and rounded to a whole bin
    because bins are ordinal.
    """
    n = int(values.shape[0])
    if kind == "bin":
        return np.full(n, float(param))
    if kind == "max":
        return np.full(n, float(max_bin))
    if kind == "quantile":
        expr = values[:, 1:]                      # exclude <cls>
        if isinstance(expr, torch.Tensor):
            expr = expr.detach().cpu().numpy()
        out = np.zeros(n, dtype=float)
        for i in range(n):
            nz = expr[i][expr[i] > 0]
            if nz.size:
                out[i] = float(np.rint(np.quantile(nz, param)))
        return out
    raise ValueError(f"Unknown target kind {kind!r}, expected one of {KINDS}")


def apply_perturbation(values, position, level, max_bin):
    """Copy of the tokenized value matrix with one gene's column overwritten.

    Returns ``(perturbed_values, fraction_of_cells_changed)``.

    The fraction is the saturation diagnostic: if a gene already sits at the
    target in most cells, the perturbation is a no-op there and the resulting
    GRN change comes only from the minority of cells with headroom.
    """
    kind, param = parse_level(level)
    out = values.clone()
    if kind == "sham":
        return out, 0.0
    targets = torch.as_tensor(
        target_bins(values, kind, param, max_bin), dtype=out.dtype
    )
    changed = float((out[:, position] != targets).to(torch.float32).mean())
    out[:, position] = targets
    return out, changed
