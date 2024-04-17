cmake_minimum_required(VERSION 3.3)

# Find ROS build system
find_package(catkin QUIET COMPONENTS roscpp rosbag sensor_msgs pcl_ros cv_bridge)

# Describe ROS project
option(ENABLE_ROS "Enable or disable building with ROS (if it is found)" ON)
if (catkin_FOUND AND ENABLE_ROS)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp rosbag sensor_msgs cv_bridge
            INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src/
            LIBRARIES check_image_lib
    )
else ()
    add_definitions(-DROS_AVAILABLE=0)
    message(WARNING "BUILDING WITHOUT ROS!")
endif ()

# Include our header files
include_directories(
        src
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        ${SOPHUS_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}      
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${SOPHUS_LIBRARIES}
        ${CERES_LIBRARIES}
        ${catkin_LIBRARIES}
)

##################################################
# Make the core library
##################################################

add_library(check_image_lib SHARED
        src/checkimgpose.cpp
        src/feat_trackandmatch.cpp
)
target_link_libraries(check_image_lib fmt::fmt ${thirdparty_libraries})
target_include_directories(check_image_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/)
install(TARGETS check_image_lib
        LIBRARY         DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME         DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

##################################################
# Make binary files!
##################################################

if (catkin_FOUND AND ENABLE_ROS)

    add_executable(run_check_image src/run_check_image.cpp)
    target_link_libraries(run_check_image check_image_lib ${thirdparty_libraries})

endif ()

