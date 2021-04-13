add_lab("ShadowRemoval")
add_lab_solution("ShadowRemoval" ${CMAKE_CURRENT_LIST_DIR}/solution.cu)

add_generator("Image" ${CMAKE_CURRENT_LIST_DIR}/image_generator.cpp)

add_lab("OtsuUnitTest")
add_lab_solution("OtsuUnitTest" ${CMAKE_CURRENT_LIST_DIR}/otsu_method/unit_test.cu)

add_lab("ColorConvertUnitTest")
add_lab_solution("ColorConvertUnitTest" ${CMAKE_CURRENT_LIST_DIR}/color_conversion/unit_test.cu)

add_lab("ConvUnitTest")
add_lab_solution("ConvUnitTest" ${CMAKE_CURRENT_LIST_DIR}/convolution/unit_test.cu)