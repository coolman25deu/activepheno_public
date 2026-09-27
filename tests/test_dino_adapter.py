import numpy as np
import pytest
import json

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from activepheno.features.image_encoder import FrozenDinoEncoder, pool_tokens


def test_patch_pooling_excludes_cls_and_registers():
    tokens = torch.tensor([[[100.], [200.], [300.], [1.], [3.]]])
    assert pool_tokens(tokens, register_tokens=2, method="patch_mean").item() == 2
    assert pool_tokens(tokens, register_tokens=2, method="cls").item() == 100
    with pytest.raises(ValueError, match="no image patch"):
        pool_tokens(tokens, register_tokens=4, method="patch_mean")


@pytest.mark.parametrize("family", ["dinov2", "dinov3"])
def test_adapter_forward_with_tiny_real_architecture(monkeypatch, family):
    if family == "dinov2":
        config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64, image_size=32, patch_size=16)
        model = transformers.Dinov2Model(config)
        processor = transformers.BitImageProcessor()
    else:
        config = transformers.DINOv3ViTConfig(hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, intermediate_size=64, image_size=32, patch_size=16,
            num_register_tokens=2)
        model = transformers.DINOv3ViTModel(config)
        processor = transformers.DINOv3ViTImageProcessorFast()
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *a, **kw: processor)
    encoder = FrozenDinoEncoder("test-only", "a" * 40)
    assert json.loads(json.dumps(encoder.descriptor)) == encoder.descriptor
    tiles = [np.zeros((32, 32, 3), dtype=np.float32), np.ones((32, 32, 3), dtype=np.float32)]
    features = encoder.encode(tiles)
    assert features.shape == (2, 32)
    assert np.isfinite(features).all()
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert not np.allclose(features[0], features[1])


def test_mutable_revision_rejected_before_loading():
    with pytest.raises(ValueError, match="immutable"):
        FrozenDinoEncoder("facebook/dinov2-base", "main")
