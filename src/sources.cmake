add_lab("ShadowRemoval")
add_lab_solution("ShadowRemoval" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)
add_generator("Image" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)

add_lab("OtsuUnitTest")
add_lab_solution("OtsuUnitTest" ${CMAKE_CURRENT_LIST_DIR}/otsu_method/unit_test.cu)