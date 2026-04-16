import torch
from fishy.models.deep.augformer import AugFormer


def test_augformer_forward():
    input_dim = 100
    output_dim = 10
    model = AugFormer(
        input_dim=input_dim, output_dim=output_dim, num_layers=2, num_heads=4
    )
    x = torch.randn(2, input_dim)
    y = model(x)
    assert y.shape == (2, output_dim)
    assert not torch.allclose(y, torch.zeros_like(y))
    print("AugFormer forward pass successful and output is not zero.")


def test_augformer_xsa():
    input_dim = 100
    output_dim = 10
    model = AugFormer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=1,
        num_heads=2,
        use_xsa=True,
    )
    x = torch.randn(2, input_dim)
    y = model(x)
    assert y.shape == (2, output_dim)
    print("AugFormer with XSA forward pass successful.")


def test_augformer_mla():
    input_dim = 100
    output_dim = 10
    model = AugFormer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=1,
        num_heads=2,
        use_mla=True,
        latent_dim=32,
    )
    x = torch.randn(2, input_dim)
    y = model(x)
    assert y.shape == (2, output_dim)
    print("AugFormer with MLA forward pass successful.")


def test_augformer_mhc():
    input_dim = 100
    output_dim = 10
    model = AugFormer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=2,
        num_heads=2,
        use_mhc=True,
    )
    x = torch.randn(2, input_dim)
    y = model(x)
    assert y.shape == (2, output_dim)
    print("AugFormer with mHC forward pass successful.")


def test_augformer_engram():
    input_dim = 100
    output_dim = 10
    model = AugFormer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=1,
        num_heads=2,
        use_engram=True,
        engram_slots=64,
    )
    x = torch.randn(2, input_dim)
    y = model(x)
    assert y.shape == (2, output_dim)
    print("AugFormer with Engram Memory forward pass successful.")


if __name__ == "__main__":
    test_augformer_forward()
    test_augformer_xsa()
    test_augformer_mla()
    test_augformer_mhc()
    test_augformer_engram()
