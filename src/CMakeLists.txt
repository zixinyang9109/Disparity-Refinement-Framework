add_library(Optical SHARED
        gaussianFilter.cpp
        ImageIO.cpp
        ImageIO.cpp
        ImagePyramid.cpp
        OpticalFlow_Huber_L1.cpp
        OpticalFlow_Huber_L2.cpp
        Stereo_Huber_L1.cpp
        Stereo_Huber_L2.cpp)

target_link_libraries(Optical
        ${THIRD_PARTY_LIBS})