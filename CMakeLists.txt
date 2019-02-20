cmake_minimum_required(VERSION 2.8.0)
project(EOFA-Grid-Quda-Test)

set ( grid_dir "/ccs/home/jiquntu/grid-build" )
set ( generic_libs "/ccs/home/jiquntu/generic-build" )

find_package(MPI)
find_package(CUDA)

set( QUDA_BUILD $ENV{HOME}/quda-build/overlapping-schwarz )
set( QUDA_SRC $ENV{HOME}/quda-src/quda )
set( QLAT_DIR $ENV{HOME}/qlat-build )

add_definitions("-O3 -fopenmp -DMPI_COMMS -DMULTI_GPU")

link_directories( ${grid_dir}/lib ${generic_libs}/lib ${QUDA_BUILD}/lib )
include_directories( ${grid_dir}/include ${generic_libs}/include ${QUDA_SRC}/include )

cuda_add_executable ( summit.x main.cc )
target_link_libraries( summit.x libGrid.a libgmp.a libmpfr.a ${MPI_LIBRARIES} rt "-fopenmp" cuda libquda.so)
