Quick Start
============

This guide is intended to jump start new Kokkos users (and beginners, in particular).


Download Latest and Build 
-----------------------------

.. note::

  Please become familiar with `Kokkos Requirements <https://kokkos.org/kokkos-core-wiki/requirements.html>`_, and verify that your machine has all necessary compilers, backend GPU SDK (e.g., CUDA, ROCM, Intel oneAPI, etc.),and build system components.


:bdg-link-primary:`Latest Release <https://github.com/kokkos/kokkos/releases/latest>`

.. code-block:: sh
  
  # Uncomment according to the type of file you've downloaded (zip or tar)
  unzip kokkos-x.y.z.zip 
  # tar -xzf kokkos-x.y.z.tar.gz
  cd kokkos-x.y.z


Basic Configure, Build, Install Recipes
----------------------------------------


OpenMP (CPU Parallelism)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

  cmake -B <build-directory> -DKokkos_ENABLE_OPENMP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install-directory> -S <source-directory>
  cmake --build <build-directory>
  cmake --install <build-directory>


.. note::

  Kokkos will attempt to autodetect GPU microarchitecture, but it is also possible to specify the desired `GPU architecture <https://kokkos.org/kokkos-core-wiki/keywords.html#gpu-architectures>`_.  In scenarios where a device (GPU) backend (e.g., CUDA, HIP) is enabled, Kokkos will default to serial execution on the host (CPU).

CUDA
~~~~

.. code-block:: sh

  cmake -B <build-directory> -DKokkos_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install-directory> -S <source-directory> 
  cmake --build <build-directory>
  cmake --install <build-directory>
  

HIP
~~~

.. code-block:: sh

  cmake -B <build-directory> -DKokkos_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install-directory> -S <source-directory>
  cmake --build <build-directory>
  cmake --install <build-directory>


Building and Linking a Kokkos "Hello World"
-------------------------------------------

.. note::

  ``Kokkos_ROOT`` and the root directory for you target backend SDK (i.e., ``CUDA_ROOT``, ``ROCM_PATH``) will need to be set.  ``Kokkos_ROOT`` should be set to the path of your Kokkos installation.  In a modules environment, the SDK variables will be typically automatically set upon module loading (e.g., ``module load rocm/5.7.1``).  Please see `Build, Install and Use <https://kokkos.org/kokkos-core-wiki/building.html>`_ for additional details.  The example detailed below is in the Kokkos Core `example` directory.



.. code-block:: sh

  git clone https://github.com/kokkos/kokkos.git 
  cd example/build_cmake_installed
  cmake -B <build-directory> -S . -DKokkos_ROOT=<install-directory>
  cd <build-directory>
  cmake --build . 
  ./example
  


Getting Help
------------

If you need addtional help getting started, please join the `Kokkos Slack Channel <https://kokkosteam.slack.com>`_.  Here are `sign up details <https://kokkos.org/kokkos-core-wiki/faq.html#faq>`_.  Joining Kokkos Slack is the on ramp for becoming a project contributor.
