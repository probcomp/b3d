#include "jax_bindings.h"
#include <tuple>
#include <pybind11/pybind11.h>

//---------------------------------------------------

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["jax_rasterize_fwd_original_gl"] = EncapsulateFunction(jax_rasterize_fwd_original_gl);
  return dict;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // State classes.
    pybind11::class_<RasterizeGLStateWrapper>(m, "RasterizeGLStateWrapper", py::module_local()).def(pybind11::init<bool, bool, int>())
        .def("set_context",     &RasterizeGLStateWrapper::setContext)
        .def("release_context", &RasterizeGLStateWrapper::releaseContext);

    // Ops.
    m.def("registrations", &Registrations, "custom call registrations");
    m.def("build_diff_rasterize_fwd_descriptor",
            [](RasterizeGLStateWrapper& stateWrapper,
            std::vector<int> all_info) {
            DiffRasterizeCustomCallDescriptor d;
            d.gl_state_wrapper = &stateWrapper;
            d.num_images = all_info[0];
            d.num_objects = all_info[1];
            d.num_vertices = all_info[2];
            d.num_triangles = all_info[3];
            d.num_layers = all_info[4];
            return PackDescriptor(d);
        });
}


//------------------------------------------------------------------------
