import torch

from torch_cpp import DCNv3, DCNv3Function, DCNv4, MSDeformAttn, dcnv3_core_pytorch


def test_dcnv3():
    # Test DCNv3
    dcnv3 = DCNv3()
    assert dcnv3 is not None
    print(dcnv3)


def test_dcnv3_forward():
    torch.manual_seed(3)

    if not torch.cuda.is_available():
        print("CUDA is not available, skipping test_dcnv3_forward")

    H_in, W_in = 8, 8
    N, M, D = 2, 4, 16
    Kh, Kw = 3, 3
    remove_center = False
    P = Kh * Kw - remove_center
    offset_scale = 2.0
    pad = 1
    dilation = 1
    stride = 1
    H_out = (H_in + 2 * pad - (dilation * (Kh - 1) + 1)) // stride + 1
    W_out = (W_in + 2 * pad - (dilation * (Kw - 1) + 1)) // stride + 1

    input = torch.rand(N, H_in, W_in, M * D).cuda() * 0.01
    offset = torch.rand(N, H_out, W_out, M * P * 2).cuda() * 10
    mask = torch.rand(N, H_out, W_out, M, P).cuda() + 1e-5
    mask /= mask.sum(-1, keepdim=True)
    mask = mask.reshape(N, H_out, W_out, M * P)

    output_pytorch = (
        dcnv3_core_pytorch(
            input,
            offset,
            mask,
            Kh,
            Kw,
            stride,
            stride,
            Kh // 2,
            Kw // 2,
            dilation,
            dilation,
            M,
            D,
            offset_scale,
            256,
            remove_center,
        )
        .detach()
        .cpu()
    )

    im2col_step = 2
    output_cuda = (
        DCNv3Function.apply(
            input,
            offset,
            mask,
            Kh,
            Kw,
            stride,
            stride,
            Kh // 2,
            Kw // 2,
            dilation,
            dilation,
            M,
            D,
            offset_scale,
            im2col_step,
            remove_center,
        )
        .detach()
        .cpu()
    )

    fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()
    print(">>> forward float")
    print(
        f"* {fwdok} check_forward_equal_with_pytorch_float: max_abs_err {max_abs_err:.2e} max_rel_err {max_rel_err:.2e}"
    )


def test_dcnv4():
    # Test DCNv4
    dcnv4 = DCNv4()
    assert dcnv4 is not None
    print(dcnv4)


def test_msdeform_attn():
    # Test MSDeformAttn
    msdeform_attn = MSDeformAttn()
    assert msdeform_attn is not None
    print(msdeform_attn)


if __name__ == "__main__":
    test_dcnv3()
    test_dcnv3_forward()
    test_dcnv4()
    test_msdeform_attn()
