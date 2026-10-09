cmake_minimum_required(VERSION 3.10)
project(yolo_onnxruntime_vendor)

option(ONNX_GPU "Use GPU for inference" OFF)

# ROS 2 package configuration
find_package(ament_cmake REQUIRED)

# Version and variant selection for the prebuilt download. All four are cache
# variables so they can be overridden with `colcon build --cmake-args`.
set(ONNXRUNTIME_VERSION "" CACHE STRING
    "ONNX Runtime version to download (default: 1.20.0 CPU; per-CUDA default for GPU)")
set(ONNX_GPU_SUFFIX "" CACHE STRING
    "GPU tarball suffix override (e.g. -gpu, -gpu_cuda12, -gpu_cuda13)")
set(ONNX_CUDA_MAJOR "" CACHE STRING
    "CUDA major version for the GPU prebuilt (11, 12 or 13); auto-detected when unset")
set(ONNXRUNTIME_URL "" CACHE STRING
    "Full ONNX Runtime tarball URL override (skips version/suffix selection)")

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
        if(NOT ARCHITECTURE STREQUAL "x64")
            message(FATAL_ERROR
                "Prebuilt GPU ONNX Runtime tarballs are x86_64-only. On "
                "${CMAKE_SYSTEM_PROCESSOR} build from source with "
                "scripts/build_ort_from_source.sh <version> --ep cuda "
                "[--cuda-arch NN], then reconfigure with -DONNXRUNTIME_ROOT=<prefix>.")
        endif()

        # The CUDA generation decides which ONNX Runtime release to use, not the
        # ROS distro. An explicit -DONNX_CUDA_MAJOR wins; otherwise detect it.
        if(NOT ONNX_CUDA_MAJOR)
            # nvidia/cuda images export CUDA_VERSION (e.g. 12.6.3).
            if(DEFINED ENV{CUDA_VERSION})
                string(REGEX MATCH "^([0-9]+)" _cuda_match "$ENV{CUDA_VERSION}")
                set(ONNX_CUDA_MAJOR "${CMAKE_MATCH_1}")
            endif()

            # A CUDA toolkit prefix exposes version.json.
            if(NOT ONNX_CUDA_MAJOR AND DEFINED ENV{CUDA_HOME} AND
               EXISTS "$ENV{CUDA_HOME}/version.json")
                file(READ "$ENV{CUDA_HOME}/version.json" _cuda_version_json)
                string(REGEX MATCH "\"version\"[ \t\r\n]*:[ \t\r\n]*\"([0-9]+)"
                       _cuda_match "${_cuda_version_json}")
                set(ONNX_CUDA_MAJOR "${CMAKE_MATCH_1}")
            endif()

            if(NOT ONNX_CUDA_MAJOR)
                file(GLOB _cuda_version_jsons "/usr/local/cuda*/version.json")
                foreach(_cuda_json IN LISTS _cuda_version_jsons)
                    file(READ "${_cuda_json}" _cuda_version_json)
                    if(_cuda_version_json MATCHES
                       "\"version\"[ \t\r\n]*:[ \t\r\n]*\"([0-9]+)\"")
                        set(ONNX_CUDA_MAJOR "${CMAKE_MATCH_1}")
                        break()
                    endif()
                endforeach()
            endif()

            if(NOT ONNX_CUDA_MAJOR)
                find_program(NVCC_EXECUTABLE nvcc
                    HINTS "$ENV{CUDA_HOME}/bin" "$ENV{CUDAToolkit_ROOT}/bin")
                if(NVCC_EXECUTABLE)
                    execute_process(COMMAND "${NVCC_EXECUTABLE}" --version
                        OUTPUT_VARIABLE _nvcc_output ERROR_QUIET)
                    string(REGEX MATCH "release ([0-9]+)" _cuda_match
                           "${_nvcc_output}")
                    set(ONNX_CUDA_MAJOR "${CMAKE_MATCH_1}")
                endif()
            endif()
        endif()

        # Curated prebuilt combinations. CUDA 11 stops shipping C++ GPU
        # tarballs after 1.18; CUDA 13 starts at 1.28. cuDNN major must match.
        if(ONNX_CUDA_MAJOR STREQUAL "11")
            set(_ort_default "1.18.0")
            set(_suffix_default "-gpu")
            set(_cudnn_default "8")
        elseif(ONNX_CUDA_MAJOR STREQUAL "12")
            set(_ort_default "1.20.0")
            set(_suffix_default "-gpu")
            set(_cudnn_default "9")
        elseif(ONNX_CUDA_MAJOR STREQUAL "13")
            set(_ort_default "1.28.0")
            set(_suffix_default "-gpu_cuda13")
            set(_cudnn_default "9")
        else()
            message(FATAL_ERROR
                "Unsupported or undetected CUDA major version "
                "('${ONNX_CUDA_MAJOR}'). Supported prebuilt combinations: "
                "11 (ORT 1.18.0), 12 (ORT 1.20.0), 13 (ORT 1.28.0). Pass "
                "-DONNX_CUDA_MAJOR=<major>, or build from source with "
                "scripts/build_ort_from_source.sh and use "
                "-DONNXRUNTIME_ROOT=<prefix>.")
        endif()

        if(NOT ONNXRUNTIME_VERSION)
            set(ONNXRUNTIME_VERSION "${_ort_default}")
        endif()

        if(NOT ONNX_GPU_SUFFIX)
            set(ONNX_GPU_SUFFIX "${_suffix_default}")
        endif()

        message(STATUS
            "ONNX Runtime GPU: CUDA ${ONNX_CUDA_MAJOR} -> ${ONNXRUNTIME_VERSION} "
            "'${ONNX_GPU_SUFFIX}' (requires cuDNN ${_cudnn_default} at run time)")
    else()
        if(NOT ONNXRUNTIME_VERSION)
            set(ONNXRUNTIME_VERSION "1.20.0")
        endif()

        if(ONNX_GPU_SUFFIX)
            message(STATUS "ONNX_GPU is OFF; ignoring ONNX_GPU_SUFFIX")
            set(ONNX_GPU_SUFFIX "")
        endif()
    endif()

    if(ONNXRUNTIME_URL)
        get_filename_component(_ort_basename "${ONNXRUNTIME_URL}" NAME_WE)
        set(ONNXRUNTIME_INSTALL_DIR "${CMAKE_BINARY_DIR}/${_ort_basename}")
        message(STATUS "Using ONNX Runtime tarball URL override: ${ONNXRUNTIME_URL}")
    else()
        set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-${ARCHITECTURE}${ONNX_GPU_SUFFIX}-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRUNTIME_INSTALL_DIR "${CMAKE_BINARY_DIR}/onnxruntime-linux-${ARCHITECTURE}${ONNX_GPU_SUFFIX}-${ONNXRUNTIME_VERSION}")
    endif()

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
