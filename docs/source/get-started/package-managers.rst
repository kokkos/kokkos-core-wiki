Package Managers
================

Use your favorite package manager to install Kokkos.

System package managers
-----------------------

DNF
~~~

You may use the Fedora Project package manager to install Kokkos
https://packages.fedoraproject.org/pkgs/kokkos/

Other package managers
----------------------

`Spack <https://spack.io>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spack is a popular package manager for HPC.  Spack comes with installation recipes for Kokkos.

https://packages.spack.io/package.html?name=kokkos summarizes the available versions of Kokkos and their options.

Most of the time, Spack Kokkos' variants follow the same options as the Kokkos `CMake options <./configuration-guide.html>`_.
List of available variants can be found by running

.. code-block::

    spack info kokkos

Installing Kokkos with Spack
+++++++++++++++++++++++++++++++

To install Kokkos with Spack with default options, run:

.. code-block::

    spack install kokkos


To install Kokkos with Cuda backend enabled, run:

.. code-block::

    spack install kokkos +cuda cuda_arch=86


Note that the `cuda_arch` option is specific to the target GPU architecture.  Here, the `cuda_arch` value `86` corresponds
to the NVIDIA Ampere architecture. With Spack, the architecture must be specified explicitly (no auto-detection).


Packaging your own Kokkos dependent project with Spack
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Let's say you have a project `Foo` that depends on `Kokkos`. You can package your project with Spack and include `Kokkos` as a dependency.

You have to create a recipe that is contained in a file named `package.py` in the `foo` directory of your recipe repository.
You can package as usual with Spack (for example, you can follow the `packaging tutorial <https://spack-tutorial.readthedocs.io/en/latest/tutorial_packaging.html>`_),
but you have to take into account that for certain backends, you might need to specify the compiler to use.

The compiler might be a wrapper, like `nvcc_wrapper` for CUDA. It is exported by the Kokkos package as the `kokkos_cxx` attribute
(Currently only available on the develop branch of Spack).

Here is an example of a `package.py` file that includes Kokkos as a dependency:

.. code-block:: python

    from spack import *

    class Foo(CMakePackage):
        homepage = "Foo"

        version('1.0', git='foo.git', tag='v1.0')

        depends_on('kokkos')

        def cmake_args(self):
            args = []
            # Ensure that the proper compiler is used
            # It might be nvcc_wrapper
            args.append(self.define("CMAKE_CXX_COMPILER", self["kokkos"].kokkos_cxx))
            return args
