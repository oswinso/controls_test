cmake_minimum_required(VERSION 3.14)
project(controls_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-W -Wall -Wextra)

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
find_package(Eigen3 3.3 REQUIRED)

set(LIBRARIES Eigen3::Eigen)

add_subdirectory(src/kinematics)
add_subdirectory(src/lqr)
add_subdirectory(src/ilqr)
add_subdirectory(src/ddp)
