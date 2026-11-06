// Copyright (c) 2026 Alejandro González Cantón
// SPDX-License-Identifier: MIT

/// @file
/// @brief ONNX Runtime C++ API include that works with both header layouts of
/// reachable through the onnxruntime::onnxruntime target.

#pragma once

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#else
#include <core/session/onnxruntime_cxx_api.h>
#endif
