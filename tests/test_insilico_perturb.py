import numpy as np
import pytest
import torch

from lib.insilico_perturb import (
    apply_perturbation,
    gene_token_position,
    level_slug,
    parse_level,
    target_bins,
)

GENES = np.array(["GATA1", "TAL1", "HBG1", "ACTB"])


def _values():
    # 3 cells x 5 tokens: [<cls>, GATA1, TAL1, HBG1, ACTB]
    return torch.tensor([[0., 5., 3., 7., 2.],
                         [0., 4., 6., 1., 9.],
                         [0., 8., 2., 5., 4.]])


# -- level parsing --------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("bin:0", ("bin", 0.0)),
    ("bin:45", ("bin", 45.0)),
    ("ko", ("bin", 0.0)),
    ("q:0.9", ("quantile", 0.9)),
    ("max", ("max", None)),
    ("sham", ("sham", None)),
])
def test_parse_level(text, expected):
    assert parse_level(text) == expected


def test_parse_level_rejects_out_of_range_quantile():
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_level("q:1.5")


def test_parse_level_rejects_garbage():
    with pytest.raises(ValueError, match="Unrecognised"):
        parse_level("2-fold")


def test_level_slug_is_filename_safe():
    assert level_slug("q:0.9") == "q0.9"
    assert level_slug("bin:0") == "bin0"


# -- positions ------------------------------------------------------------

def test_gene_position_accounts_for_the_cls_offset():
    assert gene_token_position(GENES, "GATA1") == 1
    assert gene_token_position(GENES, "ACTB") == 4


def test_unknown_gene_fails_loudly():
    with pytest.raises(KeyError, match="NOTAGENE"):
        gene_token_position(GENES, "NOTAGENE")


# -- targets --------------------------------------------------------------

def test_absolute_bin_target_is_the_same_for_every_cell():
    assert np.array_equal(target_bins(_values(), "bin", 3.0, 9.), np.full(3, 3.0))


def test_max_target_uses_the_global_max_bin():
    assert np.array_equal(target_bins(_values(), "max", None, 9.), np.full(3, 9.0))


def test_quantile_target_is_computed_per_cell():
    """q:1.0 is each cell's OWN max nonzero bin, so the target differs by cell.

    Cell nonzero values: [5,3,7,2] [4,6,1,9] [8,2,5,4] -> maxes 7, 9, 8.
    """
    assert np.array_equal(target_bins(_values(), "quantile", 1.0, 9.),
                          np.array([7., 9., 8.]))


def test_quantile_ignores_the_cls_token():
    """<cls> sits at position 0 with value 0 and must not enter the distribution."""
    v = torch.tensor([[0., 4., 4., 4., 4.]])
    assert target_bins(v, "quantile", 0.0, 9.)[0] == 4.0


def test_quantile_of_an_all_zero_cell_is_zero():
    v = torch.tensor([[0., 0., 0., 0., 0.]])
    assert target_bins(v, "quantile", 0.5, 9.)[0] == 0.0


def test_quantile_result_is_a_whole_bin():
    """Bins are ordinal; a fractional target would be off-manifold for the model."""
    out = target_bins(_values(), "quantile", 0.5, 9.)
    assert np.array_equal(out, np.rint(out))


# -- application ----------------------------------------------------------

def test_knockout_zeroes_only_the_target_column():
    out, _ = apply_perturbation(_values(), position=1, level="bin:0", max_bin=9.)
    assert torch.equal(out[:, 1], torch.zeros(3))
    assert torch.equal(out[:, 2:], _values()[:, 2:])   # neighbours untouched
    assert torch.equal(out[:, 0], _values()[:, 0])     # <cls> untouched


def test_quantile_perturbation_writes_the_per_cell_target():
    out, _ = apply_perturbation(_values(), position=1, level="q:1.0", max_bin=9.)
    assert torch.equal(out[:, 1], torch.tensor([7., 9., 8.]))


def test_perturb_returns_a_copy_not_a_view():
    v = _values()
    apply_perturbation(v, position=1, level="bin:0", max_bin=9.)
    assert v[0, 1] == 5.


def test_perturb_preserves_shape_and_dtype():
    """M must not change, or every attention rank shifts and the GRN is not
    comparable with an unperturbed run."""
    v = _values()
    out, _ = apply_perturbation(v, position=1, level="bin:0", max_bin=9.)
    assert out.shape == v.shape
    assert out.dtype == v.dtype


def test_sham_is_the_identity_and_changes_no_cells():
    """The null control: any non-zero GRN change under sham is pipeline noise."""
    v = _values()
    out, frac = apply_perturbation(v, position=1, level="sham", max_bin=9.)
    assert torch.equal(out, v)
    assert frac == 0.0


def test_saturation_fraction_counts_only_cells_that_actually_moved():
    """Cell 0 already sits at bin 5, so only 2 of 3 cells change."""
    _, frac = apply_perturbation(_values(), position=1, level="bin:5", max_bin=9.)
    assert frac == pytest.approx(2 / 3)


def test_saturation_fraction_is_zero_when_every_cell_is_already_at_target():
    v = torch.tensor([[0., 3., 1.], [0., 3., 2.]])
    _, frac = apply_perturbation(v, position=1, level="bin:3", max_bin=9.)
    assert frac == 0.0
