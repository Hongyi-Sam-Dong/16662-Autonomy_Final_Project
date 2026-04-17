cmake_minimum_required(VERSION 3.8)
project(dominoes)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetNextJointTarget.srv"
)

install(PROGRAMS
  scripts/pickup
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  dominoes
  launch
  config
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}/../franka_emika_panda
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
