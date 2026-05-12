# Forked UI: Interface/MuJoCo/simulate.cc (not mujoco::libmujoco_simulate — that embeds upstream simulate.cc).
if(NOT TARGET mujoco::platform_ui_adapter)
  message(FATAL_ERROR "MuJoCo platform_ui_adapter target was not found.")
endif()

add_library(mujoco_template_simulate STATIC
  "${CMAKE_SOURCE_DIR}/include/Interface/MuJoCo/simulate.cc")
target_include_directories(mujoco_template_simulate PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${TEMPLATE_MUJOCO_SOURCE_DIR}/simulate"
  "${TEMPLATE_MUJOCO_SOURCE_DIR}/sample"
)
target_link_libraries(mujoco_template_simulate PRIVATE
  Eigen3::Eigen
  mujoco::platform_ui_adapter
  lodepng
  mujoco::mujoco
)

if(APPLE)
  enable_language(OBJC OBJCXX)

  # GLFW's Cocoa backend is Objective-C. If CMake resolves these .m files as
  # Objective-C++, Clang rejects C-style CoreFoundation pointer conversions.
  if(TARGET glfw)
    if(CMAKE_VERSION VERSION_LESS "3.18")
      message(FATAL_ERROR "CMake 3.18 or newer is required to force GLFW Cocoa sources to Objective-C.")
    endif()

    set_source_files_properties(
      cocoa_init.m
      cocoa_joystick.m
      cocoa_monitor.m
      cocoa_window.m
      nsgl_context.m
      TARGET_DIRECTORY glfw
      PROPERTIES LANGUAGE OBJC)
  endif()

  target_sources(mujoco_template_simulate PRIVATE
    "${TEMPLATE_MUJOCO_SOURCE_DIR}/simulate/macos_gui.mm")
  target_link_libraries(mujoco_template_simulate PRIVATE "-framework Cocoa")
endif()

if(DEFINED MUJOCO_SIMULATE_COMPILE_OPTIONS)
  target_compile_options(mujoco_template_simulate PRIVATE ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
endif()
