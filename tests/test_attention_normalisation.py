"""Regression tests for the attention rank normalisation.

These drive the real ``scGPTModel.get_attention_scores`` on CPU with a stub
transformer, so the arithmetic under test is the shipped code path rather than
a copy of it.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from lib.scgpt_model import scGPTModel

D_MODEL, NHEADS, NTOK = 8, 2, 20
M = 7          # sequence length, including <cls>
N_Q = 3        # TF query positions


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.in_proj_weight = nn.Parameter(torch.randn(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = _Attn(d)

    def forward(self, x, src_key_padding_mask=None):
        return x


class _ValueEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(1, d)

    def forward(self, x):
        return self.lin(x.unsqueeze(-1))


class _Encoder(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(d) for _ in range(n)])


class _StubTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Embedding(NTOK, D_MODEL)
        self.value_encoder = _ValueEncoder(D_MODEL)
        self.transformer_encoder = _Encoder(D_MODEL, 2)


def _model():
    m = scGPTModel.__new__(scGPTModel)
    m.model = _StubTransformer().eval()
    m.nheads = NHEADS
    m.embed_size = D_MODEL
    m.device = "cpu"
    m.domain_spec_batchnorm = False
    m.num_batch_labels = None
    return m


def _inputs(n_cells=4):
    torch.manual_seed(0)
    gene_ids = torch.randint(1, NTOK, (n_cells, M))
    values = torch.rand(n_cells, M) * 5
    mask = torch.zeros(n_cells, M, dtype=torch.bool)
    conditions = np.zeros(n_cells, dtype=int)
    return gene_ids, values, mask, conditions


def _run(query_ids, n_cells=4):
    m = _model()
    gene_ids, values, mask, conditions = _inputs(n_cells)
    out = m.get_attention_scores(gene_ids, values, mask, conditions,
                                 batch_size=2, query_ids=query_ids)
    return out, n_cells


def test_output_shape_drops_the_cls_column():
    out, _ = _run(torch.tensor([1, 3, 5]))
    assert set(out) == {0}
    assert out[0].shape == (N_Q, M - 1)


def test_scores_are_finite():
    out, _ = _run(torch.tensor([1, 3, 5]))
    assert np.isfinite(out[0]).all()


def test_accumulator_is_float64_and_owns_its_memory():
    out, _ = _run(torch.tensor([1, 3, 5]))
    assert out[0].dtype == np.float64
    assert out[0].base is None


def test_column_ranks_are_normalised_by_n_q_not_M():
    """Each column is a permutation of {0, 1/n_q, ..., (n_q-1)/n_q}, so its mean
    is exactly (n_q-1)/(2*n_q). Dividing by M instead would give (n_q-1)/(2*M) --
    the bug that capped every weight at 507/4001 on the real data.
    """
    out, n_cells = _run(torch.tensor([1, 3, 5]))
    mean_matrix = out[0] / n_cells
    expected = (N_Q - 1) / (2 * N_Q)
    assert mean_matrix.mean(axis=0) == pytest.approx(expected, abs=1e-6)
    assert mean_matrix.max() <= (N_Q - 1) / N_Q + 1e-6
    # Guards against a silent regression to the old divisor.
    assert mean_matrix.mean() > (N_Q - 1) / (2 * M)


def test_every_cell_contributes_to_the_sum():
    """scGPT's notebook zeroes on first sight and only += afterwards, dropping
    the first cell of each condition. The sum must scale with the cell count."""
    one, _ = _run(torch.tensor([1, 3, 5]), n_cells=1)
    assert one[0].mean() == pytest.approx((N_Q - 1) / (2 * N_Q), abs=1e-6)


def test_square_case_still_matches_upstream_scgpt():
    """With query_ids=None, n_q == M and both passes divide by M, exactly as the
    published tutorial does."""
    m = _model()
    gene_ids, values, mask, conditions = _inputs(2)
    out = m.get_attention_scores(gene_ids, values, mask, conditions,
                                 batch_size=2, query_ids=None)
    assert out[0].shape == (M, M)      # <cls> retained, as upstream
    assert (out[0] / 2).mean(axis=0) == pytest.approx((M - 1) / (2 * M), abs=1e-6)
