cmake_minimum_required(VERSION 3.22)
project(sixf_robot)

find_package(ament_cmake REQUIRED)

# 安装Python脚本
install(PROGRAMS
  scripts/move_leg.py
  DESTINATION lib/${PROJECT_NAME}
)

# 安装launch和config等文件
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch
    DESTINATION share/${PROJECT_NAME}
    PATTERN "setup_assistant.launch" EXCLUDE)
endif()

install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

ament_package()

