import json
import types

import numpy as np
import pandas as pd
import pytest
import torch

from lib.attention_grn_process import AttentionGRNProcessor

GENES = np.array(["GATA1", "TAL1", "HBG1"])
TFS = np.array(["GATA1", "TAL1"])


def _sum_matrix():
    # 2 TFs x 3 genes, summed over 2 cells
    return np.array([[0.0, 2.0, 4.0],
                     [6.0, 8.0, 10.0]])


class _StubProcessor(AttentionGRNProcessor):
    """Exercises the orchestrator without a model, GPU, or anndata.

    Records the value matrix handed to the attention pass so a test can assert
    the perturbation actually reached the model input.
    """

    def __init__(self, values):
        super().__init__(model=None, tf_names=list(TFS))
        self._values = values
        self.seen_values = []

    def preprocess_adata(self, adata, group_key, gene_key, layer_key):
        tokenized = {
            "genes": torch.zeros(self._values.shape, dtype=torch.long),
            "values": self._values,
        }
        return (tokenized, None, np.zeros(self._values.shape[0], dtype=int),
                torch.tensor([1, 2]), {0: "K562"}, {0: 2}, GENES, TFS)

    def query_attention_score(self, tokenized_genes, tokenized_values,
                              padding_mask, condition_ids, query_ids=None,
                              batch_size=16):
        self.seen_values.append(tokenized_values.clone())
        return {0: _sum_matrix()}


def _adata(n_obs=2):
    return types.SimpleNamespace(n_obs=n_obs)


def _values():
    # 2 cells x 4 tokens: [<cls>, GATA1, TAL1, HBG1]
    return torch.tensor([[0., 5., 3., 7.],
                         [0., 4., 6., 1.]])


# -- process_grn ----------------------------------------------------------

def test_process_grn_yields_one_row_per_tf_target_pair():
    proc = AttentionGRNProcessor(model=None)
    grn = next(proc.process_grn(TFS, GENES, {0: _sum_matrix()},
                                {0: "K562"}, {0: 2}))
    assert len(grn) == 6                       # 2 TFs x 3 genes, dense
    assert list(grn.columns) == ["TF", "Target_Gene", "weight",
                                 "group_name", "num_cells"]
    assert grn["weight"].notna().all()
    assert grn.loc[0, "weight"] == pytest.approx(0.0)    # 0.0 / 2 cells


def test_process_grn_divides_by_the_cell_count():
    proc = AttentionGRNProcessor(model=None)
    grn = next(proc.process_grn(TFS, GENES, {0: _sum_matrix()},
                                {0: "K562"}, {0: 2})).set_index(["TF", "Target_Gene"])
    assert grn.loc[("TAL1", "HBG1"), "weight"] == pytest.approx(5.0)   # 10 / 2


# -- save_grn -------------------------------------------------------------

def test_save_grn_writes_a_readable_tsv(tmp_path):
    df = pd.DataFrame({"TF": ["A"], "Target_Gene": ["B"], "weight": [0.5]})
    AttentionGRNProcessor.save_grn(df, output_dir=str(tmp_path), file_name="X")
    out = tmp_path / "X_GRN.tsv"
    assert out.exists()
    assert pd.read_csv(out, sep="\t")["weight"].iloc[0] == 0.5


def test_save_grn_compresses_when_asked(tmp_path):
    df = pd.DataFrame({"TF": ["A"], "Target_Gene": ["B"], "weight": [0.5]})
    AttentionGRNProcessor.save_grn(df, output_dir=str(tmp_path), file_name="X",
                                   compress=True)
    out = tmp_path / "X_GRN.tsv.gz"
    assert out.exists()
    assert pd.read_csv(out, sep="\t")["weight"].iloc[0] == 0.5


def test_group_name_lookup_survives_a_threshold_that_drops_row_zero():
    """grn['group_name'][0] is label-based and raised KeyError once row 0 was
    filtered out; .iloc[0] is positional and does not."""
    proc = AttentionGRNProcessor(model=None)
    grn = next(proc.process_grn(TFS, GENES, {0: _sum_matrix()},
                                {0: "K562"}, {0: 2}, threshold_weight=3.0))
    assert 0 not in grn.index
    with pytest.raises(KeyError):
        _ = grn["group_name"][0]
    assert grn["group_name"].iloc[0] == "K562"


# -- process_perturbed_grn ------------------------------------------------

def test_perturbation_reaches_the_model_input(tmp_path):
    proc = _StubProcessor(_values())
    proc.process_perturbed_grn(_adata(), group_key="celltype", gene_key="index",
                               output_dir=str(tmp_path), perturb_genes=["GATA1"],
                               level="bin:0")
    handed_over = proc.seen_values[0]
    assert torch.equal(handed_over[:, 1], torch.zeros(2))   # GATA1 knocked out
    assert torch.equal(handed_over[:, 2:], _values()[:, 2:])  # others untouched


def test_perturbation_does_not_mutate_the_original_value_matrix(tmp_path):
    values = _values()
    proc = _StubProcessor(values)
    proc.process_perturbed_grn(_adata(), group_key="celltype", gene_key="index",
                               output_dir=str(tmp_path), perturb_genes=["GATA1"],
                               level="bin:0")
    assert values[0, 1] == 5.


def test_one_grn_file_per_gene_per_group_with_provenance(tmp_path):
    proc = _StubProcessor(_values())
    proc.process_perturbed_grn(_adata(), group_key="celltype", gene_key="index",
                               output_dir=str(tmp_path), perturb_genes=["GATA1"],
                               level="bin:0")
    grn = tmp_path / "GATA1_bin0_K562_GRN.tsv"
    assert grn.exists()
    df = pd.read_csv(grn, sep="\t")
    assert len(df) == 6
    assert list(df.columns) == ["TF", "Target_Gene", "weight", "group_name",
                                "num_cells", "perturbed_gene", "perturb_level"]
    assert (df["perturbed_gene"] == "GATA1").all()

    manifest = json.loads((tmp_path / "GATA1_bin0_K562_GRN.manifest.json").read_text())
    assert manifest["perturbed_gene"] == "GATA1"
    assert manifest["perturb_level"] == "bin:0"
    assert manifest["n_tfs"] == 2 and manifest["n_genes"] == 3
    assert manifest["normalisation_version"] == 2
    assert manifest["cells_changed_fraction"] == 1.0


def test_several_genes_produce_several_files(tmp_path):
    proc = _StubProcessor(_values())
    proc.process_perturbed_grn(_adata(), group_key="celltype", gene_key="index",
                               output_dir=str(tmp_path),
                               perturb_genes=["GATA1", "TAL1"], level="q:1.0")
    assert (tmp_path / "GATA1_q1.0_K562_GRN.tsv").exists()
    assert (tmp_path / "TAL1_q1.0_K562_GRN.tsv").exists()
    assert len(proc.seen_values) == 2


def test_unknown_perturb_gene_fails_before_any_gpu_work(tmp_path):
    proc = _StubProcessor(_values())
    with pytest.raises(KeyError, match="NOTAGENE"):
        proc.process_perturbed_grn(_adata(), group_key="celltype", gene_key="index",
                                   output_dir=str(tmp_path),
                                   perturb_genes=["NOTAGENE"], level="bin:0")
    assert proc.seen_values == []
