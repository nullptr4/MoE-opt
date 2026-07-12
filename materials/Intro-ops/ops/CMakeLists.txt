# Convention: operator CUDA sources auto-discovered by directory layout:
#   ops/<operator>/nvidia/*.cu  ->  compiled when CAMP_ENABLE_NVIDIA is ON
#   ops/elementwise/<operator>/nvidia/*.cu  ->  compiled when CAMP_ENABLE_NVIDIA is ON
#   ops/<operator>/metax/*.maca ->  compiled when CAMP_ENABLE_METAX is ON
#   ops/elementwise/<operator>/metax/*.maca ->  compiled when CAMP_ENABLE_METAX is ON
# After adding a new operator, re-run cmake to pick up new files.

set(CAMP_COMMON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/common/status.cc
)

set(CAMP_NVIDIA_SOURCES "")
if(CAMP_ENABLE_NVIDIA)
  file(GLOB_RECURSE CAMP_NVIDIA_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*/nvidia/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise/*/nvidia/*.cu
  )
endif()

set(CAMP_METAX_SOURCES "")
if(CAMP_ENABLE_METAX)
  file(GLOB_RECURSE CAMP_METAX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*/metax/*.maca
    ${CMAKE_CURRENT_SOURCE_DIR}/elementwise/*/metax/*.maca
  )
  set_source_files_properties(${CAMP_METAX_SOURCES} PROPERTIES LANGUAGE MACA)
endif()

set(CAMP_OPERATOR_SOURCES ${CAMP_COMMON_SOURCES} ${CAMP_NVIDIA_SOURCES} ${CAMP_METAX_SOURCES})

add_library(camp_ops SHARED ${CAMP_OPERATOR_SOURCES})
target_include_directories(camp_ops
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

if(CAMP_USE_CUTE)
  target_include_directories(camp_ops PRIVATE ${CAMP_CUTE_TARGET_INCLUDE_DIRS})
  target_compile_definitions(camp_ops PRIVATE CAMP_ENABLE_CUTE=1)
  target_compile_options(camp_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

if(CAMP_ENABLE_METAX)
  target_include_directories(camp_ops PRIVATE ${MetaX_INCLUDE_DIRS})
  target_link_libraries(camp_ops PRIVATE ${MetaX_LIBRARIES})
  target_compile_options(camp_ops PRIVATE ${CAMP_METAX_COMPILE_OPTIONS})
  set_target_properties(camp_ops PROPERTIES LINKER_LANGUAGE MACA)
endif()

target_compile_definitions(camp_ops PRIVATE
  $<$<BOOL:${CAMP_ENABLE_NVIDIA}>:CAMP_ENABLE_NVIDIA=1>
  $<$<BOOL:${CAMP_ENABLE_METAX}>:CAMP_ENABLE_METAX=1>
)

if(CAMP_ENABLE_NVIDIA)
  set_target_properties(camp_ops PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
