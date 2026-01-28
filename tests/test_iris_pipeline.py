import pytest


@pytest.mark.skip(reason="Pending: dataset-agnostic tabular pipeline for non-SMILES data (Iris).")
def test_iris_pipeline_pending():
    # Placeholder for future Iris pipeline test.
    # Expected behavior: config-driven pipeline should accept a local CSV
    # without SMILES columns and run a tabular ML flow using target_column.
    assert True
