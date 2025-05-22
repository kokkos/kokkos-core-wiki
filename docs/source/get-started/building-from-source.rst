Building From Source
====================

Getting the Kokkos Source Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section describes how to obtain the Kokkos source code.  We recommend
downloading a tagged release for most users, as these releases undergo
extensive testing and are generally more stable.  Development versions are also
available for advanced users who need the latest features and are comfortable
with potentially less stable code.

Downloading a Release Archive (Recommended)
-------------------------------------------

The recommended approach for most users is to download a release archive from GitHub.

1.  **Find the Latest Release:**  Go to the `Kokkos releases
    page <https://github.com/kokkos/kokkos/releases>`_ and find the latest
    release (or a specific version you need).

2.  **Download the Archive and Checksum:** Download both the
    ``kokkos-X.Y.Z.tar.gz`` archive and the corresponding
    ``kokkos-X.Y.Z-SHA-256.txt`` checksum file.  It's crucial to verify the
    integrity of the downloaded archive using the checksum.

3.  **Verify the Archive Integrity (Important):**  Use the following commands
    (adjust the version number as needed) to verify the downloaded archive:

    .. code-block:: sh
    
        export KOKKOS_VERSION=4.5.01  # Replace with the actual version
        export KOKKOS_DOWNLOAD_URL=https://github.com/kokkos/kokkos/releases/download/${KOKKOS_VERSION}
        curl -sLO ${KOKKOS_DOWNLOAD_URL}/kokkos-${KOKKOS_VERSION}.tar.gz
        curl -sLO ${KOKKOS_DOWNLOAD_URL}/kokkos-${KOKKOS_VERSION}-SHA-256.txt
        grep kokkos-${KOKKOS_VERSION}.tar.gz kokkos-${KOKKOS_VERSION}-SHA-256.txt | shasum -c


    The output should be ``kokkos-4.5.01.tar.gz: OK`` (or similar, depending on
    the version).  If the checksum doesn't match, **do not use the downloaded
    archive** as it may be corrupted or tampered with.

4.  **Extract the Archive:**  Once you've verified the checksum, extract the archive:

    .. code-block:: sh

        tar -xzvf kokkos-${KOKKOS_VERSION}.tar.gz

Cloning the Git Repository (For Development Versions)
-----------------------------------------------------

If you need the latest features or want to contribute to Kokkos, you can clone
the Git repository.

1.  **Clone the Repository:**

    .. code-block:: sh

        git clone https://github.com/kokkos/kokkos.git

    This will clone the repository into a directory named ``kokkos``.

2.  **Check Out a Release Tag (Recommended for Development):**
    While the ``develop`` branch is generally kept stable, it's still under
    active development.  For more predictable behavior, check out a specific
    release tag:

    .. code-block:: sh

        cd kokkos
        git checkout 4.5.01  # Replace with the desired version tag

    To see available tags:

    .. code-block:: sh

        git tag

    Or, to stay on the bleeding edge (use with caution):

    .. code-block:: sh

        git checkout develop


Which Method Should I Use?
--------------------------

* **Tagged Releases:** Use this method unless you have a specific reason to use
  a development version.  Tagged releases are the most stable and well-tested.
* **Git Repository (Development Versions):** Use this method if you need the
  very latest features, want to contribute to Kokkos, or need to debug a
  specific issue that's been fixed in the development branch.  Be aware that
  development versions may be less stable than releases.

No matter which method you choose, always verify the integrity of the
downloaded source code.  This is a crucial security practice.


Configuring and Building Kokkos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section describes how to configure and build Kokkos.  We assume you are in
the root directory of the Kokkos source code (or the project embedding Kokkos).

Configuring Kokkos
------------------

Use the following command to configure Kokkos:

.. code-block:: sh

    cmake -B builddir [<options...>]


``-B builddir`` creates a ``builddir`` directory named build (you can choose a
different name if you prefer).  Kokkos requires out-of-source builds.  The
``[<options...>]`` part is where you specify the configuration options.

**Common CMake Options**

These options are generally useful for any CMake project:

* ``-DCMAKE_CXX_COMPILER=<compiler>``: Specifies the full path to the C++
  compiler. For example, use ``hipcc`` for AMD GPUs, ``icpx`` for Intel GPUs,
  or ``g++`` or ``clang++`` for CPUs.

  Example: ``-DCMAKE_CXX_COMPILER=/path/to/hipcc``
 
* ``-DCMAKE_CXX_STANDARD=<standard>``: Sets the C++ standard. The default is ``17``.

  Example: ``-DCMAKE_CXX_STANDARD=20``

* ``-DCMAKE_BUILD_TYPE=<type>``: Controls optimization level and debugging
  information. Common options are ``Debug``, ``Release``, ``RelWithDebInfo``
  (default), and ``MinSizeRel``.

  Example: ``-DCMAKE_BUILD_TYPE=Release``

* ``-DCMAKE_INSTALL_PREFIX=<prefix>``: Specify the directory on disk to which
  Kokkos will be installed.

  Example: ``-DCMAKE_INSTALL_PREFIX=/path/to/install/dir``

**Important Kokkos-specific options:**

* ``-DKokkos_ENABLE_<BACKEND>=ON``: Enables a specific backend for target devices.
  See :ref:`keywords_backends` for a complete list, including currently open-sourced experimental backends.
  Common backends:

  * ``OPENMP`` or ``THREADS``: Multithreading on CPUs
  
  * ``CUDA``: NVIDIA GPUs
  
  * ``HIP``: AMD GPUs

  * ``SYCL``: Intel GPUs
    
  Example: ``-DKokkos_ENABLE_CUDA=ON``
  Note that ``-DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON`` is required when building with CUDA and MSVC on Windows.



  include experimental backends and :ref:`keywords_enable_backend_specific_options`.
 
 
* ``-DKokkos_ARCH_<ARCHITECTURE>=ON``: Specifies the target architecture for
  code generation. Some backends can auto-detect the architecture, but it's
  often best to specify it explicitly.
  See :ref:`keywords_arch` for a complete list.
  For instance:

  * ``AMD_GFX90A``: AMD MI210X (Frontier)

  * ``INTEL_PVC``: Intel Data Center Max 1550 (Aurora)

  * ``AMPERE80``: NVIDIA A100 (Perlmutter)

  Example: ``-DKokkos_ARCH_AMPERE80=ON``
 
* ``-DKokkos_ENABLE_DEPRECATED_CODE_4=ON``: Enables all code marked as
  deprecated. Setting this to ``OFF`` removes deprecated symbols.
  
* ``-DKokkos_ENABLE_DEPRECATION_WARNINGS=ON``: Enables deprecation warnings.
  **Strongly recommended** to avoid surprises in future releases. Don't disable
  this unless you have a very good reason.
 

**Example Configuration**

.. code-block:: sh

    cmake -B builddir \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_OPENMP=ON \
        -DKokkos_ARCH_NATIVE=ON \
        -DKokkos_ENABLE_DEPRECATED_CODE_4=OFF


Building Kokkos
---------------

After configuring, build Kokkos using:

.. code-block:: sh

    cmake --build builddir

This compiles Kokkos.  You can add ``-j<N>`` to use multiple cores for faster
compilation (replace ``<N>`` with the number of cores).

Example: ``cmake --build builddir -j8``


Installing Kokkos
-----------------

To install Kokkos (header files and libraries), use:

.. code-block:: sh

    cmake --install builddir [--prefix <prefix>]

The ``--prefix <prefix>`` option specifies the installation directory.  If
omitted, Kokkos will be installed to a default location, often ``/usr/local``
(**not recommended**).

Optional: Testing your Kokkos Build
-----------------------------------

To verify your Kokkos build and ensure everything is working as expected, you
can configure and run the internal test suite.

To do this, configure with ``-DKokkos_ENABLE_TESTS=ON``, build, and then run
the tests with:

.. code-block:: sh

    ctest --test-dir builddir --output-on-failure


Advanced: Configuring Against the Build Directory
-------------------------------------------------

(For experts only) You can configure your project directly against the
``<builddir>/cmake_packages/`` directory in the out-of-tree build, similar to
using an install tree.  This can be useful for development purposes.
