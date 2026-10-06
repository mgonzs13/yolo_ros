cmake_minimum_required(VERSION 3.8)
project(yolo_bringup)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# install the python helper package (yolo_bringup/launch_params.py) so the
# launch files can `from yolo_bringup.launch_params import ...`
ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

install(FILES
  LICENSE
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_launch_params test/test_launch_params.py)
  ament_add_pytest_test(test_launch_files test/test_launch_files.py)
endif()

ament_package()
