// Copyright (c) 2026 Miguel Ángel González Santamarta
// SPDX-License-Identifier: MIT

#include "yolo_ros/plugin/plugin_host.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/exceptions/exceptions.hpp"
#include "rclcpp/logging.hpp"

namespace yolo_ros {

PluginHost::PluginHost(rclcpp_lifecycle::LifecycleNode &node,
                       Blackboard &blackboard, TopicRegistry &topics,
                       CameraStreams &camera_streams,
                       tf2_ros::Buffer *tf_buffer, Factory factory)
    : node_(node), blackboard_(blackboard), topics_(topics),
      camera_streams_(camera_streams), tf_buffer_(tf_buffer),
      factory_(std::move(factory)) {}

PluginHost::~PluginHost() {
  if (this->activated_) {
    this->deactivate();
  }
}

std::shared_ptr<Plugin> PluginHost::create(const std::string &type,
                                           std::string &error) {
  if (this->factory_) {
    auto plugin = this->factory_(type, error);

    if (!plugin && error.empty()) {
      error = "factory returned no instance for '" + type + "'";
    }

    return plugin;
  }

  if (!this->loader_) {
    try {
      this->loader_ = std::make_unique<pluginlib::ClassLoader<Plugin>>(
          "yolo_ros", "yolo_ros::Plugin");
    } catch (const std::exception &e) {
      error = std::string("cannot create pluginlib loader: ") + e.what();
      return nullptr;
    }
  }

  try {
    return this->loader_->createSharedInstance(type);
  } catch (const std::exception &e) {
    error = "cannot load plugin '" + type + "': " + e.what();
    return nullptr;
  }
}

bool PluginHost::configure(const std::vector<std::string> &names,
                           std::string &error) {
  if (!this->instances_.empty()) {
    error = "plugins already configured";
    return false;
  }

  for (const auto &name : names) {
    if (name.empty() || name.find_first_of(":./") != std::string::npos) {
      error = "malformed plugin entry '" + name +
              "' (expected a plain instance name)";
      return false;
    }

    for (const auto &existing : this->instances_) {
      if (existing.name == name) {
        error = "duplicate plugin instance name '" + name + "'";
        return false;
      }
    }

    PluginInstance instance;
    instance.name = name;

    try {
      if (!this->node_.has_parameter(instance.name + ".plugin")) {
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        descriptor.description = "Pluginlib class for this instance";
        this->node_.declare_parameter<std::string>(instance.name + ".plugin",
                                                   "", descriptor);
      }

      instance.type =
          this->node_.get_parameter(instance.name + ".plugin").as_string();
    } catch (const std::exception &e) {
      error = std::string("plugin parameter error: ") + e.what();
      return false;
    }

    if (instance.type.empty()) {
      error = "plugin '" + instance.name + "': missing parameter '" +
              instance.name + ".plugin'";
      return false;
    }

    instance.plugin = this->create(instance.type, error);

    if (!instance.plugin) {
      return false;
    }

    this->instances_.push_back(std::move(instance));
  }

  try {
    for (auto &instance : this->instances_) {
      if (!this->node_.has_parameter(instance.name + ".cameras")) {
        this->node_.declare_parameter<std::vector<std::string>>(
            instance.name + ".cameras", std::vector<std::string>{});
      }
    }
  } catch (const std::exception &e) {
    error = std::string("plugin parameter error: ") + e.what();
    return false;
  }

  if (!this->validate_chain(error)) {
    return false;
  }

  try {
    for (auto &instance : this->instances_) {
      try {
        instance.plugin->declare_params(this->node_, instance.name + ".");
      } catch (const rclcpp::exceptions::ParameterAlreadyDeclaredException &) {
        // On Jazzy, statically typed parameters survive cleanup(), so a
        // configure retry finds them already declared. Reuse them; the
        // overrides are reapplied below so changed values still take effect.
      }
    }

    if (!this->apply_overrides(error)) {
      return false;
    }

    for (auto &instance : this->instances_) {
      instance.plugin->get_params(this->node_, instance.name + ".");
    }
  } catch (const std::exception &e) {
    error = std::string("plugin parameter error: ") + e.what();
    return false;
  }

  std::vector<std::vector<std::string>> resolved;
  resolved.reserve(this->instances_.size());

  for (std::size_t i = 0; i < this->instances_.size(); ++i) {
    std::vector<CameraInput> cameras;
    const std::vector<std::string> none;
    const auto &previous = i == 0 ? none : resolved[i - 1];

    if (!this->resolve_cameras(i, previous, cameras, error)) {
      return false;
    }

    std::vector<std::string> names;
    names.reserve(cameras.size());

    for (const auto &camera : cameras) {
      names.push_back(camera.name);
    }

    resolved.push_back(std::move(names));

    PluginContext context{this->blackboard_,        this->topics_,
                          this->node_.get_logger(), this->node_.get_clock(),
                          this->tf_buffer_,         this->instances_[i].name,
                          std::move(cameras)};

    try {
      if (!this->instances_[i].plugin->setup(context)) {
        error = "plugin '" + this->instances_[i].name + "' failed setup";
        return false;
      }
    } catch (const std::exception &e) {
      error = "plugin '" + this->instances_[i].name +
              "' exception during setup: " + e.what();
      return false;
    }
  }

  return true;
}

bool PluginHost::validate_chain(std::string &error) const {
  if (this->instances_.empty()) {
    return true;
  }

  if (this->instances_.front().type != "yolo_ros/DetectionPlugin") {
    error = "the first plugin must be yolo_ros/DetectionPlugin (got '" +
            this->instances_.front().type + "')";
    return false;
  }

  for (std::size_t i = 0; i < this->instances_.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      if (this->instances_[i].type == this->instances_[j].type) {
        error = "duplicate plugin type '" + this->instances_[i].type + "'";
        return false;
      }
    }

    if (this->instances_[i].type == "yolo_ros/DebugPlugin" &&
        i + 1 != this->instances_.size()) {
      error = "yolo_ros/DebugPlugin must be the last plugin in the chain";
      return false;
    }
  }

  return true;
}

bool PluginHost::resolve_cameras(std::size_t index,
                                 const std::vector<std::string> &previous,
                                 std::vector<CameraInput> &cameras,
                                 std::string &error) const {
  std::vector<std::string> names;
  this->node_.get_parameter(this->instances_[index].name + ".cameras", names);

  if (names.empty()) {
    for (const auto &camera : this->camera_streams_.cameras()) {
      names.push_back(camera.name);
    }
  }

  for (std::size_t i = 0; i < names.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      if (names[i] == names[j]) {
        error = "plugin '" + this->instances_[index].name +
                "': duplicate camera name '" + names[i] + "'";
        return false;
      }
    }
  }

  for (const auto &name : names) {
    const auto *config = this->camera_streams_.find(name);

    if (config == nullptr) {
      error = "plugin '" + this->instances_[index].name +
              "': unknown camera '" + name + "'";
      return false;
    }

    if (index > 0 &&
        std::find(previous.begin(), previous.end(), name) == previous.end()) {
      error = "plugin '" + this->instances_[index].name + "': camera '" + name +
              "' is not produced by plugin '" +
              this->instances_[index - 1].name + "'";
      return false;
    }

    CameraInput input;
    input.name = config->name;
    input.frame_channel = config->name;
    input.has_depth = config->has_depth();
    input.input_channel =
        index == 0
            ? config->name
            : this->instances_[index - 1].plugin->output_channel(config->name);
    cameras.push_back(std::move(input));
  }

  return true;
}

bool PluginHost::apply_overrides(std::string &error) {
  const auto &overrides =
      this->node_.get_node_parameters_interface()->get_parameter_overrides();

  if (overrides.empty()) {
    return true;
  }

  std::vector<rclcpp::Parameter> parameters;

  for (const auto &instance : this->instances_) {
    const std::string prefix = instance.name + ".";

    for (const auto &override : overrides) {
      if (override.first.compare(0, prefix.size(), prefix) == 0 &&
          this->node_.has_parameter(override.first)) {
        parameters.emplace_back(override.first, override.second);
      }
    }
  }

  if (parameters.empty()) {
    return true;
  }

  const auto results = this->node_.set_parameters(parameters);

  for (std::size_t i = 0; i < results.size(); ++i) {
    if (!results[i].successful) {
      error = "plugin parameter error: parameter '" + parameters[i].get_name() +
              "' could not be set: " + results[i].reason;
      return false;
    }
  }

  return true;
}

bool PluginHost::activate(std::string &error) {
  this->stop_ = false;
  std::size_t activated = 0;

  for (; activated < this->instances_.size(); ++activated) {
    try {
      if (!this->instances_[activated].plugin->activate()) {
        error = "plugin '" + this->instances_[activated].name +
                "' failed to activate";
        break;
      }
    } catch (const std::exception &e) {
      error = "plugin '" + this->instances_[activated].name +
              "' exception during activate: " + e.what();
      break;
    }
  }

  if (activated != this->instances_.size()) {
    // Roll back the plugin that failed activation first (it may have
    // partially acquired resources), then the successfully activated prefix
    // in reverse order, exactly like deactivate(); keep the node alive if a
    // plugin throws here.
    try {
      this->instances_[activated].plugin->deactivate();
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->node_.get_logger(),
                   "Plugin '%s' exception during deactivate: %s",
                   this->instances_[activated].name.c_str(), e.what());
    }

    for (std::size_t i = activated; i-- > 0;) {
      try {
        this->instances_[i].plugin->deactivate();
      } catch (const std::exception &e) {
        RCLCPP_ERROR(this->node_.get_logger(),
                     "Plugin '%s' exception during deactivate: %s",
                     this->instances_[i].name.c_str(), e.what());
      }
    }

    return false;
  }

  try {
    for (std::size_t i = 0; i < this->instances_.size(); ++i) {
      this->instances_[i].thread =
          std::thread([this, i] { this->run_plugin(i); });
    }
  } catch (const std::exception &e) {
    error = std::string("exception while starting plugin threads: ") + e.what();
    this->rollback_activation();
    return false;
  } catch (...) {
    error = "exception while starting plugin threads";
    this->rollback_activation();
    return false;
  }

  this->activated_ = true;
  return true;
}

void PluginHost::run_plugin(std::size_t index) {
  try {
    this->instances_[index].plugin->run(this->stop_);
  } catch (const std::exception &e) {
    RCLCPP_ERROR(this->node_.get_logger(), "Plugin '%s' crashed: %s",
                 this->instances_[index].name.c_str(), e.what());
    this->stop_ = true;
    this->blackboard_.wake_all();
    rclcpp::shutdown();
  } catch (...) {
    RCLCPP_ERROR(this->node_.get_logger(), "Plugin '%s' crashed",
                 this->instances_[index].name.c_str());
    this->stop_ = true;
    this->blackboard_.wake_all();
    rclcpp::shutdown();
  }
}

void PluginHost::rollback_activation() {
  this->stop_ = true;
  this->blackboard_.wake_all();

  for (auto &instance : this->instances_) {
    if (instance.thread.joinable()) {
      instance.thread.join();
    }
  }

  for (auto it = this->instances_.rbegin(); it != this->instances_.rend();
       ++it) {
    try {
      it->plugin->deactivate();
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->node_.get_logger(),
                   "Plugin '%s' exception during deactivate: %s",
                   it->name.c_str(), e.what());
    }
  }
}

void PluginHost::deactivate() {
  if (!this->activated_) {
    return;
  }

  this->stop_ = true;
  this->blackboard_.wake_all();

  for (auto &instance : this->instances_) {
    if (instance.thread.joinable()) {
      instance.thread.join();
    }
  }

  for (auto it = this->instances_.rbegin(); it != this->instances_.rend();
       ++it) {
    try {
      it->plugin->deactivate();
    } catch (const std::exception &e) {
      RCLCPP_ERROR(this->node_.get_logger(),
                   "Plugin '%s' exception during deactivate: %s",
                   it->name.c_str(), e.what());
    }
  }

  this->activated_ = false;
}

void PluginHost::cleanup() {
  this->deactivate();

  // Drop dynamically typed instance parameters so the next configure() can
  // declare them fresh; NodeOptions/YAML overrides persist and are reapplied
  // on the re-declaration. Statically typed parameters (declared via
  // declare_parameter<T>, as every plugin does on Jazzy) cannot be
  // undeclared; configure() reuses them and reapplies the overrides instead.
  for (const auto &instance : this->instances_) {
    const auto listed = this->node_.list_parameters({instance.name}, 0);

    for (const auto &name : listed.names) {
      try {
        this->node_.undeclare_parameter(name);
      } catch (const rclcpp::exceptions::InvalidParameterTypeException &) {
      }
    }
  }

  this->instances_.clear();
  this->loader_.reset();
}

} // namespace yolo_ros
