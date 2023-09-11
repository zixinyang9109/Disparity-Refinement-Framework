cmake_minimum_required(VERSION 3.21)
project(Optical)
#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -O2")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)


# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
        )

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
add_subdirectory(app)



#target_link_libraries(Optical ${THIRD_PARTY_LIBS})

#add_executable(Opticalflow
#        AlgorithmDriver.cpp
#        FlowToColor.cpp
#        ImageIO.cpp
#        ImageIO.h
#        ImagePyramid.cpp
#        ImagePyramid.h
#        gaussianFilter.cpp
#        gaussianFilter.h
#        OpticalFlow_Huber_L2.cpp
#        OpticalFlow_Huber_L2.h
#        )


#
#add_executable(Story
#        my_story.cpp
#        FlowToColor.cpp
#        ImageIO.cpp
#        ImageIO.h
#        ImagePyramid.cpp
#        ImagePyramid.h
#        gaussianFilter.cpp
#        gaussianFilter.h
#        OpticalFlow_Huber_L2.cpp
#        OpticalFlow_Huber_L2.h
#        )

#target_link_libraries(Story ${THIRD_PARTY_LIBS}$)
