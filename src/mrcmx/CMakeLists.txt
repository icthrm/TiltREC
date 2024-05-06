FILE(GLOB MRCMX_SRCS "*.cu" "*.cpp" "*.c")

MESSAGE(STATUS "building mrcmx")

include_directories(${PROJECT_SOURCE_DIR})

add_library (mrcmx STATIC ${MRCMX_SRCS})

#target_compile_options(mrcmx PRIVATE -O3)
target_compile_options(mrcmx PRIVATE -g)
target_link_libraries(mrcmx)

set_target_properties(mrcmx PROPERTIES OUTPUT_NAME "mrcmx")
set_target_properties(mrcmx PROPERTIES CLEAN_DIRECT_OUTPUT 1)
