cmake_minimum_required(VERSION 3.15)
project(app)

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

add_subdirectory(OpenXLSX)

add_executable(app gp.cpp)
target_link_libraries(app OpenXLSX::OpenXLSX Threads::Threads)
target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/OpenXLSX)  # Updated path