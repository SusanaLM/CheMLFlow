import pytest

from MLModels.train_models import _resolve_chemprop_foundation_config


def test_resolve_chemprop_foundation_defaults() -> None:
    foundation, checkpoint, freeze_encoder = _resolve_chemprop_foundation_config({})
    assert foundation == "none"
    assert checkpoint is None
    assert freeze_encoder is False


def test_resolve_chemprop_foundation_invalid_mode() -> None:
    with pytest.raises(ValueError, match="model.foundation"):
        _resolve_chemprop_foundation_config({"foundation": "bad_mode"})


def test_resolve_chemprop_foundation_requires_checkpoint_for_chemeleon() -> None:
    with pytest.raises(ValueError, match="foundation_checkpoint"):
        _resolve_chemprop_foundation_config({"foundation": "chemeleon"})


def test_resolve_chemprop_foundation_requires_existing_checkpoint(tmp_path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(ValueError, match="does not exist"):
        _resolve_chemprop_foundation_config(
            {"foundation": "chemeleon", "foundation_checkpoint": str(missing)}
        )


def test_resolve_chemprop_foundation_with_checkpoint_and_freeze(tmp_path) -> None:
    ckpt = tmp_path / "chemeleon_mp.pt"
    ckpt.write_bytes(b"placeholder")
    foundation, checkpoint, freeze_encoder = _resolve_chemprop_foundation_config(
        {
            "foundation": "chemeleon",
            "foundation_checkpoint": str(ckpt),
            "freeze_encoder": True,
        }
    )
    assert foundation == "chemeleon"
    assert checkpoint == str(ckpt)
    assert freeze_encoder is True


def test_resolve_chemprop_foundation_freeze_requires_chemeleon() -> None:
    with pytest.raises(ValueError, match="freeze_encoder"):
        _resolve_chemprop_foundation_config({"freeze_encoder": True})
