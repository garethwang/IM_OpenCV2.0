cmake_minimum_required(VERSION 2.8) 

project(DEMO_IM) 

if (CMAKE_COMPILER_IS_GNUCXX)

	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")

endif (CMAKE_COMPILER_IS_GNUCXX)

add_definitions(-DSOURCE_DIR="${CMAKE_SOURCE_DIR}")

find_package(OpenCV 2.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.h ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(demo_im demo_im.cpp ${SRC_FILES})

target_link_libraries(demo_im ${OpenCV_LIBS})

