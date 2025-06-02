add_executable(Story my_story.cpp)
target_link_libraries(Story Optical ${THIRD_PARTY_LIBS})

add_executable(DeformableIGS DeformableIGS.cpp)
target_link_libraries(DeformableIGS Optical ${THIRD_PARTY_LIBS})

add_executable(AlgorithmDriver AlgorithmDriver.cpp)
target_link_libraries(AlgorithmDriver Optical ${THIRD_PARTY_LIBS})

add_executable(StageOne Stage1.cpp)
target_link_libraries(StageOne Optical ${THIRD_PARTY_LIBS})

add_executable(StageTwo Stage2.cpp)
target_link_libraries(StageTwo Optical ${THIRD_PARTY_LIBS})

add_executable(StageLTwo StageL2.cpp)
target_link_libraries(StageLTwo Optical ${THIRD_PARTY_LIBS})

add_executable(TestMed test_median_filter.cpp)
target_link_libraries(TestMed Optical ${THIRD_PARTY_LIBS})

add_executable(Albation Albation.cpp)
target_link_libraries(Albation Optical ${THIRD_PARTY_LIBS})

add_executable(SGM_post sgm_post.cpp)
target_link_libraries(SGM_post Optical ${THIRD_PARTY_LIBS})