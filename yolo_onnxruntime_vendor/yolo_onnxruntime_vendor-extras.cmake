# Installed ONNX Runtime (AMENT_VENDOR_POLICY=NEVER_VENDOR*): expose it.
find_package(onnxruntime CONFIG REQUIRED)

list(APPEND yolo_onnxruntime_vendor_LIBRARIES onnxruntime::onnxruntime)
