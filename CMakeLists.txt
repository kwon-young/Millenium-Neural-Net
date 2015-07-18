
cmake_minimum_required(VERSION 3.1.3)
project(Millenium-Neural-Net C CXX)

set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

include_directories ("${PROJECT_SOURCE_DIR}/include")

file( GLOB_RECURSE source_files "src/*")

add_executable(${PROJECT_NAME} ${source_files})

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_SOURCE_DIR})

add_custom_target(run
  COMMAND make install
  COMMAND ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}
  DEPENDS ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME}
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
