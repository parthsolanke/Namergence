import importlib
import sys
from pathlib import Path

import pytest
import torch

SRC_DIR = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def server_module(monkeypatch):
    monkeypatch.chdir(SRC_DIR)
    sys.modules.pop("server", None)

    module = importlib.import_module("server")

    yield module

    sys.modules.pop("server", None)


def test_category_from_output_maps_index_to_label(server_module):
    output = torch.full((1, len(server_module.loaded_labels)), -10.0)
    output[0][2] = 0.0

    label = server_module.category_from_output(output)

    assert label == server_module.loaded_labels[2]


def test_predict_returns_label_and_confidence_for_every_category(server_module):
    response = server_module.predict(server_module.InputName(name="Alice"))

    assert response["name"] == "Alice"
    assert response["label"] in response["categories"]
    assert len(response["categories"]) == len(server_module.loaded_labels)

    confidences = response["confidences"][0]
    assert len(confidences) == len(response["categories"])
    assert abs(sum(confidences) - 1.0) < 1e-5
