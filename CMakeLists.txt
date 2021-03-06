# ----------------------------------------------------------------------------
#  Root CMake file for Synthetic Blur Project
# ----------------------------------------------------------------------------
#
# Example CMake build line for windows & VS2015
# windows: cmake -G "Visual Studio 14 2015 Win64" -DUSE_AVX_INSTRUCTIONS=ON ..
# linux: cmake -DUSE_AVX_INSTRUCTIONS=ON ..
#
#
# To compile:
# windows: cmake --build . --config Release
# linux: cmake --build . --config Release -- -j10
#

cmake_minimum_required(VERSION 2.8.12)

PROJECT(Blur_Project)


set(exe_name "blur_gen")

# This looks for an environment variable called PLATFORM.  This can be used to point
# cmake to various locations based on the individual machines that the code is complied on
set(PLATFORM "$ENV{PLATFORM}" CACHE INTERNAL "Copied from environment variable")
message("Platform: " ${PLATFORM})


if(${PLATFORM} STREQUAL "Laptop")
    set(COMMON_ROOT "D:/Common")
    set(DFD_COMMON_ROOT "D:/Projects/dfd_common")
elseif(${PLATFORM} STREQUAL "MainGear")
    set(COMMON_ROOT "~/Common")
    set(DFD_COMMON_ROOT "~/Projects/dfd_common")
elseif(${PLATFORM} STREQUAL "LaptopN")
    set(COMMON_ROOT "C:/Projects/Common")
    set(DFD_COMMON_ROOT "C:/Projects/dfd_common")
elseif(${PLATFORM} STREQUAL "HPC")
    set(COMMON_ROOT "$HOME/Common")
    set(DFD_COMMON_ROOT "$HOME/Projects/dfd_common")
else()
    set(COMMON_ROOT "D:/Common")
    set(DFD_COMMON_ROOT "D:/Projects/dfd_common")	
endif()

include(${COMMON_ROOT}/cmake_utils/use_cpp_11.cmake)
include(${COMMON_ROOT}/cmake_utils/use_sse_avx.cmake)

# grab all of the *.c and *.cpp files in the src directory
file(GLOB SOURCES "src/*.c*")

set(project_includes "include" "${COMMON_ROOT}/include" "${DFD_COMMON_ROOT}/include")

# Display the source files found
message(" ")
message("--------------------------------------------------------")
message("Blur Generation Project Source Files: ")
foreach(S ${SOURCES})
    message("  " ${S})
endforeach()
message("--------------------------------------------------------")
message(" ")


# find OpenCV
find_package(OpenCV 4 REQUIRED)
if (OpenCV_FOUND)
	set(project_includes ${project_includes} ${OpenCV_INCLUDE_DIRS})
    
    message(" ")
    message("--------------------------------------------------------")
    message("Blur Generation Project Includes: ")
    foreach(S ${project_includes})
        message("  " ${S})
    endforeach()
    message("--------------------------------------------------------")
    message(" ")
    
    include_directories(${project_includes})
	add_executable(${exe_name} ${SOURCES})
	target_link_libraries(${exe_name} ${OpenCV_LIBS})

else()
	message("OpenCV not found, so we won't build the project.")
endif()

if(WIN32)
    # uncomment for MSVC and cmake greater than 3.6.0
    set(default_build_type "Release")
    if(MSVC AND (${CMAKE_VERSION} VERSION_GREATER "3.6.0"))
        set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${exe_name})
    endif()

endif()

message(" ")
message(STATUS "Build Complete!")
message(" ")
