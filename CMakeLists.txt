cmake_minimum_required(VERSION 3.15)
project(mienc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add custom flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -Wextra -std=c++1z")


# Add the C++ subdir
add_subdirectory(cpp)

# Install Python package files
install(
    DIRECTORY src/mienc/
    DESTINATION mienc
    FILES_MATCHING PATTERN "*.py"
)