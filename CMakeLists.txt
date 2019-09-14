cmake_minimum_required(VERSION 2.8) 
set(targetname Demo_IM) 
project(${targetname}) 

if (CMAKE_COMPILER_IS_GNUCXX)

	set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")

endif (CMAKE_COMPILER_IS_GNUCXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.h ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(demo_im demo_im.cpp ${SRC_FILES})

target_link_libraries(demo_im ${OpenCV_LIBS})

