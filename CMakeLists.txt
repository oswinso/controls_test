cmake_minimum_required(VERSION 3.14)
project(controls_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-W -Wall -Wextra)

include_directories(include)
find_package(Eigen3 3.3 REQUIRED)

set(LIBRARIES Eigen3::Eigen)

add_subdirectory(src/lqr)
