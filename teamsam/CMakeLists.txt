cmake_minimum_required(VERSION 3.0.2)
project(dominoes)

find_package(catkin REQUIRED COMPONENTS
  rospy
  message_generation
)

catkin_python_setup()

add_service_files(
  FILES
  GetNextJointTarget.srv
)

generate_messages()

catkin_package(
  CATKIN_DEPENDS rospy message_runtime
)

catkin_install_python(PROGRAMS
  scripts/pickup
  scripts/pickup_client
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../franka_emika_panda
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
