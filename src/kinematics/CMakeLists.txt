add_library(kinematics INTERFACE)
target_include_directories(kinematics INTERFACE ${INCLUDE_DIR})
target_link_libraries(kinematics INTERFACE ${LIBRARIES})
