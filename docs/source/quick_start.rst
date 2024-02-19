============
Quick Start
============

  This guide is intended to get new Kokkos users started in less than one hour.  Kokkos Core is the foundation for all other Kokkos Ecosystem projects.

------------
Requirements
------------

::

  CMake >=3.18
  gcc >= 8.3.1
  CUDA >= 11
  rocm >=5.7

--------------------------------------
Download Current Release, Set Up Build 
--------------------------------------

..
  Choice 1 for obtaining current release:
..

* `Click to Download Current Release <https://github.com/kokkos/kokkos/archive/refs/heads/master.zip>`_ 


..
  Temp. Notes --
  Buttons, badges, icons:
  https://sphinx-design.readthedocs.io/en/latest/badges_buttons.html
  :bdg-link-primary-line:`explicit title <https://example.com>`

  https://getbootstrap.com/docs/5.0/components/badge/
..


..
  Choice 2 for obtaining current release:
..

:bdg-link-primary:`Current Release <https://github.com/kokkos/kokkos/archive/refs/heads/master.zip>`


::

  unzip kokkos-master.zip
  cd kokkos-master
  mkdir build
  cd build


----------------------------------------------
Basic Configuration Examples for Source Builds
----------------------------------------------

  You can create small shell scripts to manage configuration details, following the GPU microarchitecture-appropriate examples below.  Upon successful configuration, ``make -j`` to build, and ``make install`` to install.

Serial
~~~~~~
::

  export INSTALL_DIR = ${MY_INSTALL_DIR}

  cmake \
    -DKokkos_ARCH_NATIVE=ON \
    -DKokkos_ENABLE_OPENMP=OFF \
    -DKokkos_ENABLE_SERIAL=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=Release ..


OpenMP
~~~~~~
::

  export INSTALL_DIR = ${MY_INSTALL_DIR}

  cmake \
   -DKokkos_ARCH_NATIVE=ON \
   -DKokkos_ENABLE_OPENMP=ON \
   -DKokkos_ENABLE_SERIAL=ON \
   -DCMAKE_CXX_STANDARD=17 \
   -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
   -DCMAKE_BUILD_TYPE=Release ..


CUDA
~~~~

::

  export INSTALL_DIR = ${MY_INSTALL_DIR}

  cmake \
    -DKokkos_ARCH_NATIVE=ON \
    -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=Release ..

.. note::

  VEGA90A is for AMD MI200
  AMD_GFX942 is for AMD MI300

HIP
~~~
::

  cmake \
    -DCRAYPE_LINK_TYPE=dynamic \
    -DKokkos_ENABLE_TESTS=ON \
    -DKokkos_ENABLE_HIP=ON \
    -DKokkos_ARCH_AMD_GFX942=OFF \
    -DKokkos_ARCH_VEGA90A=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_COMPILER=$(which hipcc) \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
    -DCMAKE_BUILD_TYPE=Release ..

----------------------------------------------
Basic Configuration Examples for Spack Builds
----------------------------------------------

*  A number of Kokkos variants can be built / installed via Spack.  You will need to select a variant for the desired backend, and appropriate GPU microarchitecture.   
*  To explore the range of variants for a package, ``spack info kokkos``, ``spack info trilinos``, etc.
* `Basic Spack Installation <https://spack.readthedocs.io/en/latest/getting_started.html>`_



.. note::

  Before installing, you can ``spack spec``  variants to verify the build type.

Serial
~~~~~~~

::

  spack spec kokkos@4.2 %gcc@10.3.0 +serial cxxstd=20

OpenMP
~~~~~~

::

  spack spec kokkos@4.2 %gcc@10.3.0 +openmp cxxstd=20


CUDA
~~~~

:: 
  
  spack spec / install kokkos@4.2 %gcc@12.2.0 +cuda cuda_arch=70 cxxstd=20 +cuda_relocatable_device_code


HIP
~~~

::

  spack spec / install kokkos@4.2 %gcc@12.2.0 +rocm amdgpu_target=gfx90a cxxstd=20


--------------------
Additional Resources
--------------------

* `CMake Keywords <https://kokkos.org/kokkos-core-wiki/keywords.html>`_
* `Building Kokkos <https://kokkos.org/kokkos-core-wiki/building.html>`_
* `Spack Kokkos Build <https://kokkos.org/kokkos-core-wiki/building.html#spack>`_


---
FAQ
---

* How to Join Kokkos Slack?

You can find the Slack channel at `kokkosteam.slack.com <https://kokkosteam.slack.com>`_. Register a new account with your email. We automatically whitelist emails from most organizations, but if your email address is not whitelisted, you can contact the Kokkos maintainers (their emails are in the LICENSE file).


* How to build for more recent C++??

When configuring Kokkos with CMake (see configruation, add the flag ``-DCMAKE_CXX_STANDARD=20`` (or ``23``). Ensure that the flag is also set for any downstream applications.

