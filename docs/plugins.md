# Writing a plugin

Pipeline stages are **pluginlib** plugins. To add your own:

1. Derive from `yolo_ros::Plugin` and implement `declare_params`, `get_params`, `output_channel`, `setup`, `activate`, `deactivate` and `run` (the worker loop, which must poll its `stop` argument and return promptly). The two parameter methods receive the owning lifecycle node and the instance `prefix`, a runtime string ending in a dot (e.g. `"detection."`), so a plugin declares and reads every parameter under its `<instance>.*` namespace without hard-coding the instance name:

   ```cpp
   void declare_params(rclcpp_lifecycle::LifecycleNode &node,
                       const std::string &prefix) override {
     node.declare_parameter<double>(prefix + "inference.threshold", 0.7);
   }

   void get_params(const rclcpp_lifecycle::LifecycleNode &node,
                   const std::string &prefix) override {
     node.get_parameter(prefix + "inference.threshold", this->threshold_);
   }
   ```

2. Implement `output_channel(camera)` returning the per-camera channel/topic your plugin produces (the chain wires it as the next plugin's input). In `setup`, read the cameras to process from `ctx.cameras` — each `CameraInput` carries the camera name, the internal frame channel and the upstream `input_channel` — and declare/expose outputs with `ctx.blackboard.declare_channel<T>(channel)` and `ctx.topics.expose<T>(channel, topic, qos, ctx.name)`. Plugin inputs are always in-process channels wired by the node; `TopicRegistry` only publishes.
3. Register the class with `PLUGINLIB_EXPORT_CLASS(my_ns::MyPlugin, yolo_ros::Plugin)` and add a matching `<class>` entry to [`yolo_ros/plugins.xml`](../yolo_ros/plugins.xml).
4. Load it by adding `"<instance>"` to the node's `plugins` parameter (respecting the chain rules: detection first, no duplicate types, debug last) and setting `<instance>.plugin: yolo_ros/MyPlugin` plus an optional `<instance>.cameras` list; its parameters are then declared and read under `<instance>.*`.

See `yolo_ros/src/plugins/debug_plugin.cpp` for a complete example.
