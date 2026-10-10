cmake_minimum_required(VERSION 3.14)
project(rover_control)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake     REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp          REQUIRED)
find_package(std_msgs        REQUIRED)
find_package(geometry_msgs   REQUIRED)
find_package(nav_msgs        REQUIRED)
find_package(yaml-cpp        REQUIRED)

include_directories(include)

# odom_node
add_executable(odom_node src/odom_node.cpp)
ament_target_dependencies(odom_node
  rclcpp std_msgs nav_msgs geometry_msgs ament_index_cpp
)
target_link_libraries(odom_node
  yaml-cpp
  ament_index_cpp::ament_index_cpp
)

# lqr_node
add_executable(lqr_node src/lqr_node.cpp)
ament_target_dependencies(lqr_node
  rclcpp std_msgs nav_msgs geometry_msgs ament_index_cpp
)
target_link_libraries(lqr_node
  yaml-cpp
  ament_index_cpp::ament_index_cpp
)

install(TARGETS odom_node lqr_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
