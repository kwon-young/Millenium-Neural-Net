
cmake_minimum_required(VERSION 3.1.3)
project(Millenium-Neural-Net C CXX)

option (USE_MKL
  "Use Intel Math Kernel Library" ON)

set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

include_directories ("${PROJECT_SOURCE_DIR}/include")

file( GLOB_RECURSE source_files "src/*")

if (USE_MKL)
  set(MKL_ROOT "C:/Prog/Intel/Composer XE 2015/mkl" CACHE STRING "MKL library root directory")
  add_definitions(-DEIGEN_USE_MKL_ALL)
  include_directories ("${MKL_ROOT}/include")
  set (EXTRA_LIBS ${EXTRA_LIBS} mkl_rt)
  LINK_DIRECTORIES(${MKL_ROOT}/lib/mic)
endif (USE_MKL)

add_executable(${PROJECT_NAME} ${source_files})

target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS})

add_custom_command( TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Copying MNIST data.."
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/data ${CMAKE_BINARY_DIR}
  )

#INSTALL(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_SOURCE_DIR})

add_custom_target(run
  COMMAND ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME}
  DEPENDS ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME}
  WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
