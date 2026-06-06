import importlib
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeJaxtype:
    def __getitem__(self, _item):
        return object


def import_evaluate_loss_with_dataset_stub(load_dataset):
    jaxtyping_module = types.ModuleType("jaxtyping")
    jaxtyping_module.Float = FakeJaxtype()
    jaxtyping_module.Int = FakeJaxtype()
    sys.modules["jaxtyping"] = jaxtyping_module

    datasets_module = types.ModuleType("datasets")
    datasets_module.load_dataset = load_dataset
    sys.modules["datasets"] = datasets_module

    model_base_module = types.ModuleType("pipeline.model_utils.model_base")
    model_base_module.ModelBase = object
    sys.modules["pipeline.model_utils.model_base"] = model_base_module

    for module_name in [
        "pipeline.submodules.evaluate_loss",
        "pipeline.utils.hook_utils",
    ]:
        sys.modules.pop(module_name, None)

    return importlib.import_module("pipeline.submodules.evaluate_loss")


class FakeTokenizer:
    def __call__(self, texts, return_tensors, padding, truncation, max_length):
        assert texts == ["first pile row", "second pile row"]
        assert return_tensors == "pt"
        assert padding is True
        assert truncation is True
        assert max_length == 8
        return {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.int64),
        }


def test_batch_iterator_pile_loads_streaming_dataset_without_remote_code_flag():
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return iter([
            {"text": "first pile row"},
            {"text": "second pile row"},
        ])

    evaluate_loss = import_evaluate_loss_with_dataset_stub(fake_load_dataset)

    inputs, loss_mask = next(evaluate_loss.batch_iterator_pile(FakeTokenizer(), batch_size=2, max_length=8))

    assert calls == [
        (
            ("monology/pile-uncopyrighted",),
            {"split": "train", "streaming": True},
        )
    ]
    assert "trust_remote_code" not in calls[0][1]
    assert inputs["input_ids"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert loss_mask.tolist() == [[1, 1, 0], [1, 1, 0]]
