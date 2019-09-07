add_library(ddp INTERFACE)
target_include_directories(ddp INTERFACE ${INCLUDE_DIR})
target_link_libraries(ddp INTERFACE ${LIBRARIES} lqr)

add_executable(ddp_demo main.cpp)
target_link_libraries(ddp_demo ${LIBRARIES} ddp kinematics)
