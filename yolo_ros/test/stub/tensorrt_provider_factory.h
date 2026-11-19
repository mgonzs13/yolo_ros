// Copyright (C) 2026 Miguel Ángel González Santamarta
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
