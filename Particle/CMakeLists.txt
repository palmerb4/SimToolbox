add_executable(ParticleNear_test ParticleNear_test.cpp)
target_compile_options(ParticleNear_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(ParticleNear_test PRIVATE ${PROJECT_SOURCE_DIR}
                                                     ${Trilinos_INCLUDE_DIRS})
target_link_libraries(
  ParticleNear_test 
  PRIVATE ${Trilinos_LIBRARIES} 
          ${Trilinos_TPL_LIBRARIES}
          Eigen3::Eigen 
          OpenMP::OpenMP_CXX 
          MPI::MPI_CXX)
add_test(NAME SylidnerNear COMMAND ParticleNear_test)

add_executable(
  ParticleSystem_main
  ParticleSystem_main.cpp
  ParticleConfig.cpp
  ParticleShapes.cpp
  ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp
  ${PROJECT_SOURCE_DIR}/Boundary/Boundary.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/BCQPSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintCollector.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintOperator.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintSolver.cpp
  ${PROJECT_SOURCE_DIR}/Util/Base64.cpp)
target_compile_options(ParticleSystem_main PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_definitions(
  ParticleSystem_main PRIVATE PARTICLE_SIMULATOR_THREAD_PARALLEL
                              PARTICLE_SIMULATOR_MPI_PARALLEL)
target_include_directories(
  ParticleSystem_main PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                              ${TRNG_INCLUDE_DIR} ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(
  ParticleSystem_main
  PRIVATE ${Trilinos_LIBRARIES}
          ${Trilinos_TPL_LIBRARIES}
          ${YAML_CPP_LIBRARIES}
          ${TRNG_LIBRARY}
          ${Boost_LIBRARIES}
          VTK::IOXML
          Eigen3::Eigen
          OpenMP::OpenMP_CXX
          MPI::MPI_CXX)

add_executable(
  ParticleSystem_test_api
  ParticleSystem_test_api.cpp
  ParticleConfig.cpp
  ParticleShapes.cpp
  ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp
  ${PROJECT_SOURCE_DIR}/Boundary/Boundary.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/BCQPSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintCollector.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintOperator.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintSolver.cpp
  ${PROJECT_SOURCE_DIR}/Util/Base64.cpp)
target_compile_options(ParticleSystem_test_api PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_definitions(
  ParticleSystem_test_api PRIVATE PARTICLE_SIMULATOR_THREAD_PARALLEL
                                  PARTICLE_SIMULATOR_MPI_PARALLEL)
target_include_directories(
  ParticleSystem_test_api
  PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE_DIR}
          ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(
  ParticleSystem_test_api
  PRIVATE ${Trilinos_LIBRARIES}
          ${Trilinos_TPL_LIBRARIES}
          ${YAML_CPP_LIBRARIES}
          ${TRNG_LIBRARY}
          ${Boost_LIBRARIES}
          VTK::IOXML
          Eigen3::Eigen
          OpenMP::OpenMP_CXX
          MPI::MPI_CXX)

# Scan through resource folder for updated files and copy if none existing or changed
file(GLOB_RECURSE resources "./Test*/*.*")
foreach(resource ${resources})
  get_filename_component(filename ${resource} NAME)
  get_filename_component(dir ${resource} DIRECTORY)
  get_filename_component(dirname ${dir} NAME)

  set(output "${CMAKE_CURRENT_BINARY_DIR}/TestCases/${dirname}/${filename}")

  add_custom_command(
    COMMENT "Moving updated resource-file '${resource}'"
    OUTPUT ${output}
    DEPENDS ${resource}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${resource} ${output})
  add_custom_target(${dirname}_${filename} ALL DEPENDS ${resource} ${output})

endforeach()

add_test(
  NAME Confinement
  COMMAND
    sh -c "cd TestCases/Test1_Confinement/ \
    && export OMP_NUM_THREADS=2 \
    && mpirun -n 2 ../../ParticleSystem_main > Confinement.log \
    && python Verify.py")
set_tests_properties(Confinement PROPERTIES FAIL_REGULAR_EXPRESSION
                                            "[^a-z]Error;ERROR;Failed")

add_test(
  NAME MixLink
  COMMAND
    sh -c "cd TestCases/Test2_MixLink/ \
    && export OMP_NUM_THREADS=2 \
    && mpirun -n 2 ../../ParticleSystem_main > MixLink.log \
    && python Verify.py")
set_tests_properties(MixLink PROPERTIES FAIL_REGULAR_EXPRESSION
                                        "[^a-z]Error;ERROR;Failed")

add_test(
  NAME TestAPI
  COMMAND
    sh -c "cd TestCases/Test3_TestAPI/ \
    && export OMP_NUM_THREADS=1 \
    && mpirun -n 4 ../../ParticleSystem_test_api > TestAPI.log \
    && python Verify.py")
set_tests_properties(TestAPI PROPERTIES FAIL_REGULAR_EXPRESSION
                                        "[^a-z]Error;ERROR;Failed")

add_test(
  NAME StressLC
  COMMAND
    sh -c "cd TestCases/Test4_StressLC/ \
    && export OMP_NUM_THREADS=3 \
    && mpirun -n 2 ../../ParticleSystem_main > StressLC.log \
    && python StressVerify.py")
set_tests_properties(StressLC PROPERTIES FAIL_REGULAR_EXPRESSION
                                         "[^a-z]Error;ERROR;Failed")

add_test(
  NAME StressSphere
  COMMAND
    sh -c "cd TestCases/Test5_StressSphere/ \
    && export OMP_NUM_THREADS=2 \
    && mpirun -n 4 ../../ParticleSystem_main > StressSphere.log \
    && python StressVerify.py")
set_tests_properties(StressSphere PROPERTIES FAIL_REGULAR_EXPRESSION
                                             "[^a-z]Error;ERROR;Failed")
