// Copyright (C) 2026 Alejandro González Cantón
// Portions Copyright (C) 2024 Miguel Ángel González Santamarta
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

#include "yolo_ros/utils/logs.hpp"

#include <cstdarg>
#include <cstdio>

namespace {

/// @brief Print "[LEVEL] [file:function:line] message" to stderr.
void stderr_vlog(const char *level, const char *file, const char *function,
                 int line, const char *text, va_list args) {
  std::fprintf(stderr, "[%s] [%s:%s:%d] ", level, file, function, line);
  std::vfprintf(stderr, text, args);
  std::fprintf(stderr, "\n");
}

void stderr_log_error(const char *file, const char *function, int line,
                      const char *text, ...) {
  va_list args;
  va_start(args, text);
  stderr_vlog("ERROR", file, function, line, text, args);
  va_end(args);
}

void stderr_log_warn(const char *file, const char *function, int line,
                     const char *text, ...) {
  va_list args;
  va_start(args, text);
  stderr_vlog("WARN", file, function, line, text, args);
  va_end(args);
}

void stderr_log_info(const char *file, const char *function, int line,
                     const char *text, ...) {
  va_list args;
  va_start(args, text);
  stderr_vlog("INFO", file, function, line, text, args);
  va_end(args);
}

void stderr_log_debug(const char *file, const char *function, int line,
                      const char *text, ...) {
  va_list args;
  va_start(args, text);
  stderr_vlog("DEBUG", file, function, line, text, args);
  va_end(args);
}

} // namespace

namespace yolo_ros::utils {

LogFunction log_error = stderr_log_error;
LogFunction log_warn = stderr_log_warn;
LogFunction log_info = stderr_log_info;
LogFunction log_debug = stderr_log_debug;

std::atomic<LogLevel> log_level = INFO;

void set_log_level(LogLevel level) { log_level = level; }

} // namespace yolo_ros::utils
