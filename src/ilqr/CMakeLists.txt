add_library(ilqr INTERFACE)
target_include_directories(ilqr INTERFACE ${INCLUDE_DIR})
target_link_libraries(ilqr INTERFACE ${LIBRARIES} lqr)

add_executable(ilqr_demo main.cpp)
target_link_libraries(ilqr_demo ${LIBRARIES} ilqr kinematics)
