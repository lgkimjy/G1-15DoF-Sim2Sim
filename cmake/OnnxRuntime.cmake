set(SIM2SIM_ONNXRUNTIME_SOURCE_DIR "${CMAKE_SOURCE_DIR}/include/3rd-parties/onnxruntime")

find_path(ONNXRUNTIME_INCLUDE_DIR
  onnxruntime_cxx_api.h
  HINTS "${SIM2SIM_ONNXRUNTIME_SOURCE_DIR}"
  PATH_SUFFIXES include include/onnxruntime
  NO_DEFAULT_PATH
)
find_library(ONNXRUNTIME_LIBRARY
  onnxruntime
  HINTS "${SIM2SIM_ONNXRUNTIME_SOURCE_DIR}"
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH
)

set(_onnxruntime_has_headers FALSE)
find_path(ONNXRUNTIME_HEADER_ONLY_HINT
  onnxruntime_cxx_api.h
  HINTS "${SIM2SIM_ONNXRUNTIME_SOURCE_DIR}"
  PATH_SUFFIXES
    include
    include/onnxruntime
    include/onnxruntime/core/session
  NO_DEFAULT_PATH
)
if(ONNXRUNTIME_HEADER_ONLY_HINT)
  set(_onnxruntime_has_headers TRUE)
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  if(_onnxruntime_has_headers AND NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
      "ONNX Runtime headers were found, but the runtime library was not.\n"
      "It looks like include/3rd-parties/onnxruntime contains the source repo only.\n"
      "Place a built/prebuilt ONNX Runtime there so this exists too:\n"
      "  include/3rd-parties/onnxruntime/lib/libonnxruntime.*"
    )
  else()
    message(FATAL_ERROR
      "ONNX Runtime not found.\n"
      "Place a built/prebuilt ONNX Runtime in:\n"
      "  include/3rd-parties/onnxruntime\n"
      "Example : (mac) from onnxruntime github, download release file, \n"
      "  and tar -xzf onnxruntime-osx-arm64-1.25.0.tgz \n"
      "  and mv to include/3rd-parties/onnxruntime\n"
      "Example : (linux) from onnxruntime github, download release file, \n"
      "  and tar -xzf onnxruntime-linux-x64-gpu-1.25.0.tgz \n"
      "  and mv to include/3rd-parties/onnxruntime\n"
      "Expected layout:\n"
      "  include/3rd-parties/onnxruntime/include/...\n"
      "  include/3rd-parties/onnxruntime/lib/libonnxruntime.*"
    )
  endif()
endif()

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

get_filename_component(_ort_lib_dir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
if(APPLE)
  set_property(TARGET onnxruntime::onnxruntime APPEND PROPERTY INTERFACE_LINK_OPTIONS "LINKER:-rpath,${_ort_lib_dir}")
endif()
