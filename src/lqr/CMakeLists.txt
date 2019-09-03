add_library(lqr INTERFACE)
target_include_directories(lqr INTERFACE ${INCLUDE_DIR})
target_link_libraries(lqr INTERFACE ${LIBRARIES})

add_executable(lqr_demo main.cpp)
target_link_libraries(lqr_demo lqr kinematics)
