cmake_minimum_required(VERSION 3.8)
project(yolo_onnxruntime_vendor)

option(ONNX_GPU "Use GPU for inference" OFF)

# ROS 2 package configuration
find_package(ament_cmake REQUIRED)

set(ONNXRUNTIME_VERSION "1.20.0")  # version of the prebuilt download

if(ONNXRUNTIME_ROOT)
    # A locally built ONNX Runtime prefix (scripts/build_ort_from_source.sh), in the
    # flat lib/ + include/ layout. Takes precedence over the prebuilt download.
    if(NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
        message(FATAL_ERROR
            "ONNXRUNTIME_ROOT='${ONNXRUNTIME_ROOT}' has no lib/libonnxruntime.so. "
            "Build it with scripts/build_ort_from_source.sh <version> first.")
    endif()
    set(ONNXRUNTIME_INSTALL_DIR "${ONNXRUNTIME_ROOT}")
    message(STATUS "Using locally built ONNX Runtime at ${ONNXRUNTIME_INSTALL_DIR}")
else()
    # Determine the architecture of the prebuilt tarball
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(ARCHITECTURE "x64")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(ARCHITECTURE "aarch64")
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    if(ONNX_GPU)
        set(GPU "-gpu")
    else()
        set(GPU "")
    endif()

    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ARCHITECTURE}${GPU}-${ONNXRUNTIME_VERSION}.tgz")
    set(ONNXRUNTIME_INSTALL_DIR "${CMAKE_BINARY_DIR}/onnxruntime-linux-${ARCHITECTURE}${GPU}-${ONNXRUNTIME_VERSION}")

    # Add a custom target to download and extract the prebuilt ONNX Runtime
    find_program(CURL_EXECUTABLE curl REQUIRED)

    if(NOT CURL_EXECUTABLE)
        message(FATAL_ERROR "curl is required to download ONNX Runtime but was not found.")
    endif()

    # The extracted shared library acts as the build output marker: CMake only
    # runs the download+extract commands when it is missing, so a clean rebuild
    # does not re-download the ~200 MB tarball every time.
    set(ONNXRUNTIME_TGZ "${CMAKE_BINARY_DIR}/onnxruntime.tgz")
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_INSTALL_DIR}/lib/libonnxruntime.so")

    add_custom_command(
        OUTPUT ${ONNXRUNTIME_LIB}
        COMMAND ${CURL_EXECUTABLE} -L -o ${ONNXRUNTIME_TGZ} ${ONNXRUNTIME_URL}
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${ONNXRUNTIME_TGZ}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Downloading and extracting ONNX Runtime ${ONNXRUNTIME_VERSION}"
        VERBATIM
    )

    add_custom_target(download_onnxruntime ALL
        DEPENDS ${ONNXRUNTIME_LIB}
    )
endif()

# Install the ONNX Runtime library and include files
install(DIRECTORY ${ONNXRUNTIME_INSTALL_DIR}/lib DESTINATION .)
install(DIRECTORY ${ONNXRUNTIME_INSTALL_DIR}/include DESTINATION .)

install(FILES
    LICENSE
    THIRD_PARTY_NOTICES.md
    DESTINATION share/${PROJECT_NAME}
)

# Preserve the notices shipped with the ONNX Runtime distribution, including
# licenses for its bundled third-party components.
install(FILES
    ${ONNXRUNTIME_INSTALL_DIR}/LICENSE
    ${ONNXRUNTIME_INSTALL_DIR}/ThirdPartyNotices.txt
    DESTINATION share/${PROJECT_NAME}/onnxruntime
)

# Export the onnxruntime library for downstream packages
ament_export_include_directories(include)
ament_export_libraries(onnxruntime)

# Export the package
ament_package()
