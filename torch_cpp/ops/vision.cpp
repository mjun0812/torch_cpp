#include "dcnv3.h"
#include "dcnv4.h"
#include "ms_deform_attn.h"

namespace torch_cpp {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // DCNv3
  m.def("dcnv3_forward", &dcnv3_forward, "dcnv3_forward");
  m.def("dcnv3_backward", &dcnv3_backward, "dcnv3_backward");

  // DCNv4
  m.def("flash_deform_attn_forward", &flash_deform_attn_forward,
        "flash_deform_attn_forward");
  m.def("flash_deform_attn_backward", &flash_deform_attn_backward,
        "flash_deform_attn_backward");
  m.def("dcnv4_forward", &dcnv4_forward, "dcnv4_forward");
  m.def("dcnv4_backward", &dcnv4_backward, "dcnv4_backward");

  // Multiscale Deformable Attention
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
}  // namespace torch_cpp
