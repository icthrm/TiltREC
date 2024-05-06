
FILE(GLOB OPTS_SRCS "*.cu" "*.cpp" "*.c")

MESSAGE(STATUS "building opts")

include_directories(${PROJECT_SOURCE_DIR})

add_library (opts STATIC ${OPTS_SRCS})

target_link_libraries(opts)

set_target_properties(opts PROPERTIES OUTPUT_NAME "opts")
set_target_properties(opts PROPERTIES CLEAN_DIRECT_OUTPUT 1)
