// Copyright (c) 2026 Alejandro González Cantón
// Portions Copyright (c) 2024 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

/// @file
/// @brief ROS-free, printf-style logging used by the inference engine.
///
/// The engine layer must not depend on ROS, so it logs through the function
/// pointers below. They default to stderr in the
/// "[LEVEL] [file:function:line] message" format; a host application (e.g. a
/// ROS node) may swap them for its own sinks before loading a model.

#ifndef YOLO_ROS__UTILS__LOGS_HPP_
#define YOLO_ROS__UTILS__LOGS_HPP_

#include <atomic>
#include <cstring>

/// @addtogroup yolo_utils
/// @{
namespace yolo_ros::utils {

/// @brief Signature of a printf-style logging sink.
/// @param file Source file (the macros pass a basename).
/// @param function Enclosing function name.
/// @param line Source line.
/// @param text printf-style format string.
using LogFunction = void (*)(const char *file, const char *function, int line,
                             const char *text, ...);

/// @brief Sink used by YOLO_LOG_ERROR. Defaults to stderr.
extern LogFunction log_error;
/// @brief Sink used by YOLO_LOG_WARN. Defaults to stderr.
extern LogFunction log_warn;
/// @brief Sink used by YOLO_LOG_INFO. Defaults to stderr.
extern LogFunction log_info;
/// @brief Sink used by YOLO_LOG_DEBUG. Defaults to stderr.
extern LogFunction log_debug;

/// @brief Severity levels, most to least severe.
enum LogLevel {
  ERROR = 0, ///< Errors only.
  WARN,      ///< Warnings and above.
  INFO,      ///< Informational messages and above (default).
  DEBUG      ///< Everything.
};

/// @brief Current verbosity; the macros skip messages below it.
extern std::atomic<LogLevel> log_level;

/// @brief Basename of @p path (handles '/' and '\\').
/// @param[in] path Source path.
/// @return Pointer inside @p path at the file name.
inline const char *extract_filename(const char *path) {
  const char *filename = std::strrchr(path, '/');

  if (!filename) {
    filename = std::strrchr(path, '\\');
  }

  return filename ? filename + 1 : path;
}

/// @brief Set the minimum severity emitted by the YOLO_LOG_* macros.
/// @param[in] level New level.
void set_log_level(LogLevel level);

} // namespace yolo_ros::utils
/// @}

/// @brief Log at ERROR level (always emitted).
#define YOLO_LOG_ERROR(text, ...)                                              \
  do {                                                                         \
    if (yolo_ros::utils::log_level.load() >= yolo_ros::utils::ERROR) {         \
      yolo_ros::utils::log_error(yolo_ros::utils::extract_filename(__FILE__),  \
                                 __func__, __LINE__, text, ##__VA_ARGS__);     \
    }                                                                          \
  } while (0)

/// @brief Log at WARN level.
#define YOLO_LOG_WARN(text, ...)                                               \
  do {                                                                         \
    if (yolo_ros::utils::log_level.load() >= yolo_ros::utils::WARN) {          \
      yolo_ros::utils::log_warn(yolo_ros::utils::extract_filename(__FILE__),   \
                                __func__, __LINE__, text, ##__VA_ARGS__);      \
    }                                                                          \
  } while (0)

/// @brief Log at INFO level (default verbosity).
#define YOLO_LOG_INFO(text, ...)                                               \
  do {                                                                         \
    if (yolo_ros::utils::log_level.load() >= yolo_ros::utils::INFO) {          \
      yolo_ros::utils::log_info(yolo_ros::utils::extract_filename(__FILE__),   \
                                __func__, __LINE__, text, ##__VA_ARGS__);      \
    }                                                                          \
  } while (0)

/// @brief Log at DEBUG level.
#define YOLO_LOG_DEBUG(text, ...)                                              \
  do {                                                                         \
    if (yolo_ros::utils::log_level.load() >= yolo_ros::utils::DEBUG) {         \
      yolo_ros::utils::log_debug(yolo_ros::utils::extract_filename(__FILE__),  \
                                 __func__, __LINE__, text, ##__VA_ARGS__);     \
    }                                                                          \
  } while (0)

#endif // YOLO_ROS__UTILS__LOGS_HPP_
