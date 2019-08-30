#add_library(lqr lqr.cpp)
#target_link_libraries(lqr ${LIBRARIES})
#
add_executable(lqr_demo main.cpp)
target_link_libraries(lqr_demo ${LIBRARIES})
