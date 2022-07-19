Requirements
############

Compiler Versions
=================

Generally Kokkos should work with all compiler versions newer than the minimum.
However,in complex code, we have to work around compiler bugs. So compiler versions we don't test may have issues
we are unaware of.

.. list-table::
    :widths: 30 35 35
    :header-rows: 1
    :align: center

    * - Compiler
      - Minimum version
      - Primary tested versions

    * * GCC 
      * 5.3.0
      * 5.3.0, 6.1.0, 7.3.0, 8.3, 9.2, 10.0
    
    * * Clang 
      * 4.0.0
      * 8.0.0, 9.0.0, 10.0.0, 12.0.0
    
    * * Intel 
      * 17.0.1
      * 17.4, 18.1, 19.5
    
    * * NVCC 
      * 9.2.88
      * 9.2.88, 10.1, 11.0
    
    * * NVC++ 
      * 21.5
      * NA
    
    * * ROCM 
      * 4.5
      * 4.5.0
    
    * * MSVC 
      * 19.29
      * 19.29
    
    * * IBM XL 
      * 16.1.1
      * 16.1.1
    
    * * Fujitsu 
      * 4.5.0
      * NA
    
    * * ARM/Clang 
      * 20.1
      * 20.1

Build system:
=============

* CMake >= 3.16: required
* CMake >= 3.18: Fortran linkage. This does not affect most mixed Fortran/Kokkos builds. See [build issues](BUILD.md#KnownIssues).
* CMake >= 3.21.1 for NVC++

Primary tested compiler are passing in release mode
with warnings as errors. They also are tested with a comprehensive set of
backend combinations (i.e. OpenMP, Pthreads, Serial, OpenMP+Serial, ...).
We are using the following set of flags:

* GCC:

.. code-block:: bash

  -Wall -Wunused-parameter -Wshadow -pedantic
  -Werror -Wsign-compare -Wtype-limits
  -Wignored-qualifiers -Wempty-body
  -Wclobbered -Wuninitialized

* Intel:

.. code-block:: bash

  -Wall -Wunused-parameter -Wshadow -pedantic
  -Werror -Wsign-compare -Wtype-limits
  -Wuninitialized

* Clang:

.. code-block:: bash

  -Wall -Wunused-parameter -Wshadow -pedantic
  -Werror -Wsign-compare -Wtype-limits
  -Wuninitialized

* NVCC:

.. code-block:: bash

  -Wall -Wunused-parameter -Wshadow -pedantic
  -Werror -Wsign-compare -Wtype-limits
  -Wuninitialized

.. note:: 

  Other compilers are tested occasionally, in particular when pushing from develop to master branch. These are tested less rigorously without ``-Werror`` and only for a select set of backends.
