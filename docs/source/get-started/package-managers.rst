Package Managers
================

Use your favorite package manager to install Kokkos.

System package managers
~~~~~~~~~~~~~~~~~~~~~~~

+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+
| Distro   | Command to install                    | Backend              | Vetted | Maintainer                            | Build Source                                                                                             |
+==========+=======================================+======================+========+=======================================+==========================================================================================================+
| Fedora   | ``dnf install kokkos-devel``          | openMP, rocm, serial | a bit  | rbberger                              | `here <https://src.fedoraproject.org/rpms/kokkos/blob/rawhide/f/kokkos.spec>`_                           |
+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+
| Debian   | ``apt-get install libkokkos-dev``     | openMP               | a bit  | alexmyczko                            | `here <https://salsa.debian.org/debian/kokkos/-/blob/master/debian/rules>`_                              |
+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+
| Conda    | ``conda install conda-forge::kokkos`` | openMP (linux), Threads (windows), serial, CUDA | nope   | carterbox                             | `here <https://github.com/conda-forge/kokkos-feedstock/blob/main/recipe/build.sh>`_                      |
+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+
| openSUSE | ``zypper install kokkos-devel``       | openMP, serial       | nope   | `mli@suse.com <mailto:mli@suse.com>`_ | `here <https://build.opensuse.org/projects/science/packages/kokkos/files/kokkos.spec?expand=1>`_         |
+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+
| Gentoo   | ``emerge kokkos``                     | whatever enabled     | a bit  | tamiko                                | `here <https://gitweb.gentoo.org/repo/gentoo.git/tree/dev-cpp/kokkos/kokkos-4.3.1.ebuild>`_              |
+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+
| Arch     | ``pacman -S kokkos``                  | Threads, serial      | maybe  | carlosal1015                          | `here <https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=kokkos>`_                                  |
+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+
| Mac Port | ``port install kokkos-devel``         | openMP, serial       | no     | MarcusCalhoun-Lopez                   | `here <https://github.com/macports/macports-ports/blob/master/devel/kokkos/Portfile>`_                   |
+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+
| Spack    | ``spack install kokkos``              | whatever enabled     | yes    | cedricchevalier19, nmm0, lucbv        | `here <https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/kokkos/package.py>`_ |
+----------+---------------------------------------+----------------------+--------+---------------------------------------+----------------------------------------------------------------------------------------------------------+

Other package managers
~~~~~~~~~~~~~~~~~~~~~~

`Spack <https://spack.io>`_
---------------------------

Spack is a popular package manager for HPC.  Spack comes with installation recipes for Kokkos.

The `Kokkos recipe webpage <https://packages.spack.io/package.html?name=kokkos>`_ summarizes the available versions of Kokkos
and their options.

Most of the time, Spack Kokkos' variants follow the same options as the Kokkos `CMake options <./configuration-guide.html>`_.
List of available variants can be found by running

.. code-block::

    spack info kokkos


When using Spack, Kokkos hardware autodetection is disabled. That means that the user always has to manually specify the 
architecture. However, for CPU, Spack already specify the CPU micro-architecture, so it is not needed to specify it again.
For GPU, no such mechanism exists in Spack and the user always need to specify the correct architecture, using a dedicated
backend keyword (see next section).


Installing Kokkos with Spack
++++++++++++++++++++++++++++

To install Kokkos with Spack with default options, run:

.. code-block::

    spack install kokkos


To install Kokkos with CUDA backend enabled, run:

.. code-block::

    spack install kokkos +cuda cuda_arch=90


Note that the `cuda_arch` option is specific to the target GPU architecture.  Here, the `cuda_arch` value `90` corresponds
to the NVIDIA Hopper architecture. With Spack, the architecture must be specified explicitly (no auto-detection).


For AMD GPU, the traditional Spack's keyword is `rocm` instead of `hip` in Kokkos' CMake. So to install Kokkos with the HIP backend enable, run:

.. code-block::

    spack install kokkos +rocm amdgpu_target=gfx942


Note that the `amdgpu_target` option is specific to the target GPU architecture.
With Spack, the architecture must be specified explicitly (no auto-detection).


For Intel GPU, using the SYCL backend, run:

.. code-block::

    spack spec kokkos +sycl intel_gpu_arch=intel_pvc


Note that the `intel_gpu_arch` option is specific to the target GPU architecture.
With Spack, the architecture must be specified explicitly (no auto-detection).


To use the installed Kokkos, you can simply load the Kokkos module:

.. code-block::

    spack load kokkos


This will inject the Kokkos environment into your shell session.

Packaging your own Kokkos dependent project with Spack
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Let's say you have a project `Foo` that depends on `Kokkos`. You can package your project with Spack and include `Kokkos` as a dependency.

You have to create a recipe that is contained in a file named `package.py` in the `foo` directory of your recipe repository.
You can package as usual with Spack (for example, you can follow the `packaging tutorial <https://spack-tutorial.readthedocs.io/en/latest/tutorial_packaging.html>`_),
but you have to take into account that for certain backends, you might need to specify the compiler to use.

The compiler might be a wrapper, like `nvcc_wrapper` for CUDA. It is exported by the Kokkos package as the `kokkos_cxx` attribute.

Here is an example of a `package.py` file that includes Kokkos as a dependency:

.. code-block:: python

    from spack.package import *

    class Foo(CMakePackage):
        # Usual description of a Spack package

        depends_on("kokkos")

        def cmake_args(self):
            args = []
            # Ensure that the proper compiler is used
            # It might be nvcc_wrapper
            args.append(self.define("CMAKE_CXX_COMPILER", self["kokkos"].kokkos_cxx))
            return args


For more complete examples, you can look at already existing recipes in the *Required by* section of
`Kokkos Spack recipe <https://packages.spack.io/package.html?name=kokkos>`_ or by running:

.. code-block::

    spack dependents kokkos


`Conda <https://https://anaconda.org/>`_
----------------------------------------

You may use the Conda package manager to install Kokkos:

.. code-block::

    conda install conda-forge::kokkos

