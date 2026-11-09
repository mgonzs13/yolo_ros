# Installed ONNX Runtime (AMENT_VENDOR_POLICY=NEVER_VENDOR*): expose it through
# the same variables as the vendored build.
find_package(onnxruntime CONFIG REQUIRED)

get_target_property(_yolo_ort_includes onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
foreach(_dir IN LISTS _yolo_ort_includes)
  list(APPEND yolo_onnxruntime_vendor_INCLUDE_DIRS "${_dir}")
  # conda-forge's onnxruntime-cpp keeps the public headers in the source tree
  # layout (core/session), while yolo_ros uses #include <onnxruntime_cxx_api.h>.
  # This workaround can be removed after: https://github.com/conda-forge/onnxruntime-feedstock/pull/221
  if(EXISTS "${_dir}/core/session/onnxruntime_cxx_api.h")
    list(APPEND yolo_onnxruntime_vendor_INCLUDE_DIRS "${_dir}/core/session")
  endif()
endforeach()
unset(_yolo_ort_includes)

list(APPEND yolo_onnxruntime_vendor_LIBRARIES onnxruntime::onnxruntime)
