// Test stub for the ORT 1.6 TensorRT provider factory header. It exists only
// so the legacy TRT branch can be compiled on a host without TensorRT.
#ifndef YOLO_ROS__TEST__STUB__TENSORRT_PROVIDER_FACTORY_H_
#define YOLO_ROS__TEST__STUB__TENSORRT_PROVIDER_FACTORY_H_

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tensorrt,
               _In_ OrtSessionOptions *options, int device_id);

#ifdef __cplusplus
}
#endif

#endif // YOLO_ROS__TEST__STUB__TENSORRT_PROVIDER_FACTORY_H_
