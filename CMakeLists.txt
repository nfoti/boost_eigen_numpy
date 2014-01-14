cmake_minimum_required(VERSION 2.8)
project(np_test)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED OFF)
set(Boost_USE_STATIC_RUNTIME OFF)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)

find_package(PythonLibs 2.7 REQUIRED)
find_package(NumPy REQUIRED)
find_package(Eigen3 REQUIRED)

message("Boost_INCLUDE_DIRS: " ${Boost_INCLUDE_DIRS})
message("Boost_PYTHON_INCLUDE_DIR: " ${Boost_PYTHON_INCLUDE_DIR})
message("Boost libraries: " ${Boost_LIBRARIES})
message("PYTHON_INCLUDE_DIRS: " ${PYTHON_INCLUDE_DIRS})
message("EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${BOOST_PYTHON_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

add_library(np_test SHARED np_test.cpp)
set_target_properties(np_test PROPERTIES PREFIX "" SUFFIX ".so")
target_link_libraries(np_test ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
