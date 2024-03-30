from torch_cpp import DCNv3, DCNv4, MSDeformAttn


def test_dcnv3():
    # Test DCNv3
    dcnv3 = DCNv3()
    assert dcnv3 is not None


def test_dcnv4():
    # Test DCNv4
    dcnv4 = DCNv4()
    assert dcnv4 is not None


def test_msdeform_attn():
    # Test MSDeformAttn
    msdeform_attn = MSDeformAttn()
    assert msdeform_attn is not None


if __name__ == "__main__":
    test_dcnv3()
    test_dcnv4()
    test_msdeform_attn()
