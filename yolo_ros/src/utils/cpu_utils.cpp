// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/utils/cpu_utils.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace yolo_ros::utils {

namespace {

#if defined(__linux__)
/// @brief Parse a Linux CPU list such as "0-3,5,7-8" into CPU ids.
/// @param list CPU list string.
/// @return The CPU ids; empty when malformed.
std::vector<int> parse_cpu_list(const std::string &list) {
  std::vector<int> cpus;
  std::stringstream stream(list);
  std::string token;

  while (std::getline(stream, token, ',')) {
    const auto dash = token.find('-');

    try {
      if (dash == std::string::npos) {
        if (!token.empty()) {
          cpus.push_back(std::stoi(token));
        }
        continue;
      }

      const int first = std::stoi(token.substr(0, dash));
      const int last = std::stoi(token.substr(dash + 1));

      for (int cpu = first; cpu <= last; ++cpu) {
        cpus.push_back(cpu);
      }
    } catch (const std::exception &) {
      // Ignore malformed entries.
    }
  }

  return cpus;
}

/// @brief Read the first line of @p path (empty on failure).
/// @param path File to read.
/// @return The first line, or an empty string.
std::string read_first_line(const std::string &path) {
  std::ifstream file(path);
  std::string line;

  if (file && std::getline(file, line)) {
    return line;
  }

  return "";
}
#endif

} // namespace

int available_cpus() {
#if defined(__linux__)
  cpu_set_t affinity;

  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
    const int count = CPU_COUNT(&affinity);

    if (count > 0) {
      return count;
    }
  }
#endif

  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware > 0 ? static_cast<int>(hardware) : 1;
}

int num_physical_cores() {
#if defined(__linux__)
  const std::vector<int> online =
      parse_cpu_list(read_first_line("/sys/devices/system/cpu/online"));
  std::set<std::pair<int, int>> cores;

  for (const int cpu : online) {
    const std::string base =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    const std::string package = read_first_line(base + "physical_package_id");
    const std::string core = read_first_line(base + "core_id");

    if (package.empty() || core.empty()) {
      continue;
    }

    try {
      cores.emplace(std::stoi(package), std::stoi(core));
    } catch (const std::exception &) {
      // Ignore malformed entries.
    }
  }

  if (!cores.empty()) {
    return static_cast<int>(cores.size());
  }
#endif

  return available_cpus();
}

int num_math_threads() {
  const int logical = available_cpus();

#if defined(__linux__)
  // Intel hybrid CPUs expose the performance cores through cpu_core; math work
  // runs there, like llama.cpp's hybrid handling.
  const std::vector<int> performance =
      parse_cpu_list(read_first_line("/sys/devices/cpu_core/cpus"));

  if (!performance.empty()) {
    return std::max(1, std::min(static_cast<int>(performance.size()), logical));
  }
#endif

  return std::max(1, std::min(num_physical_cores(), logical));
}

} // namespace yolo_ros::utils
