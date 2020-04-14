import pytest
import torch

from dyna_bert.models.heads import ClassificationHead


@pytest.mark.parametrize("batch_size, hidden_size, num_class", [pytest.param(1, 3, 12), pytest.param(3, 12, 4)])
def test_shape_of_classification_heads(batch_size: int, hidden_size: int, num_class: int):
    """Check shape of Classification head model outputs"""
    classification_head = ClassificationHead(hidden_size, num_class)

    pooled_output = torch.rand((batch_size, hidden_size))
    outputs = classification_head(pooled_output)

    assert outputs.shape == (batch_size, num_class)