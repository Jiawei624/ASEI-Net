import ast
import inspect
from pathlib import Path

import pytest
import torch

from asei.compat import load_torch_checkpoint, transformer_encoder
from asei.config import ROOT
from scripts.check_environment import requirements, inspect_packages, satisfies


def test_legacy_torch_load_signature(tmp_path, monkeypatch):
    path = tmp_path / "weights.pt"
    torch.save({"weights": torch.ones(2)}, path)
    original = torch.load
    restricted = "weights_only" in inspect.signature(original).parameters
    calls = []

    def legacy_load(file, map_location=None):
        calls.append(map_location)
        kwargs = {"weights_only": True} if restricted else {}
        return original(file, map_location=map_location, **kwargs)

    monkeypatch.setattr(torch, "load", legacy_load)
    checkpoint = load_torch_checkpoint(path, map_location="cpu")
    assert calls == ["cpu"]
    torch.testing.assert_close(checkpoint["weights"], torch.ones(2))


def test_restricted_load_failure_is_not_retried(monkeypatch):
    calls = []

    def current_load(file, map_location=None, *, weights_only=False):
        calls.append(weights_only)
        raise RuntimeError("Invalid checkpoint")

    monkeypatch.setattr(torch, "load", current_load)
    with pytest.raises(RuntimeError, match="Invalid checkpoint"):
        load_torch_checkpoint("unused.pt")
    assert calls == [True]


def test_legacy_transformer_constructor(monkeypatch):
    original = torch.nn.TransformerEncoder
    has_nested = "enable_nested_tensor" in inspect.signature(original).parameters

    def legacy_encoder(layer, layers, norm=None):
        kwargs = {"enable_nested_tensor": False} if has_nested else {}
        return original(layer, layers, norm=norm, **kwargs)

    monkeypatch.setattr(torch.nn, "TransformerEncoder", legacy_encoder)
    layer = torch.nn.TransformerEncoderLayer(8, 2, batch_first=True, norm_first=True)
    encoder = transformer_encoder(layer, 1, torch.nn.LayerNorm(8))
    output = encoder(torch.randn(2, 4, 8), mask=torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1))
    assert output.shape == (2, 4, 8) and torch.isfinite(output).all()


def test_python38_source_grammar():
    for folder in ("asei", "scripts"):
        for path in (ROOT / folder).glob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=8)


def test_environment_reuses_tab_versions_and_reports_only_missing_yaml(monkeypatch):
    from importlib import metadata
    installed = {"numpy": "1.21.0", "pandas": "1.3.5", "torch": "1.11.0+cpu", "scikit-learn": "1.0.2",
                 "scipy": "1.7.3", "statsmodels": "0.14.0", "tqdm": "4.64.0"}

    def version(name):
        if name not in installed:
            raise metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(metadata, "version", version)
    rules = requirements(ROOT / "requirements.txt")
    rows = inspect_packages(rules, requirements(ROOT / "third_party" / "TAB" / "requirements.txt"))
    assert [row["package"] for row in rows if row["status"] != "ok"] == ["PyYAML"]
    assert installed["torch"] == "1.11.0+cpu" and installed["numpy"] == "1.21.0"


def test_strict_numpy_profile_does_not_treat_new_numpy_as_exact_tab_pin(monkeypatch):
    from importlib import metadata
    monkeypatch.setattr(metadata, "version", lambda name: "2.2.6")
    reuse = inspect_packages({"numpy": requirements(ROOT / "requirements.txt")["numpy"]}, {})
    strict = inspect_packages({"numpy": requirements(ROOT / "requirements-tab.txt")["numpy"]}, {})
    assert reuse[0]["status"] == "ok" and strict[0]["status"] == "incompatible"


def test_package_version_local_build_and_prerelease_rules():
    assert satisfies("1.11.0+cpu", ">=", "1.11.0")
    assert not satisfies("1.11.0rc1", ">=", "1.11.0")
    assert satisfies("1.21.0", "==", "1.21.0")
    assert not satisfies("1.21.6", "==", "1.21.0")
