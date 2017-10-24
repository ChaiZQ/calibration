cmake_minimum_required(VERSION 2.8)
set( CMAKE_CXX_FLAGS "-std=c++11" )

#opencv
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

#self-defined library
add_library(eyeHand src/eyeHand.cc)

add_executable( calibration src/calibration.cc)
target_link_libraries( calibration ${OpenCV_LIBS})
target_link_libraries( eyeHand ${OpenCV_LIBS})
target_link_libraries(calibration eyeHand)
