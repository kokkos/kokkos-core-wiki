Macros
======

Including ``<Kokkos_Macros.hpp>`` (which ``<Kokkos_Core.hpp>`` does) gives access
to two kinds of macros.

Always-defined macros
---------------------

These macros are defined by every Kokkos build. They are meant to be used directly in application code,
to mark functions, to select host or device code, or to check the Kokkos version.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 70

   * - Page
     - Content
   * - :doc:`Function annotations <macros-special/host_device_macros>`
     - ``KOKKOS_FUNCTION``, ``KOKKOS_LAMBDA``, etc., mark functions that can be called from the host and/or the device.
   * - :doc:`Conditional host/device code <macros-special/if_on_host_or_device>`
     - ``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE`` select code depending on whether it is compiled for the host or the device.
   * - :doc:`Version <macros/version>`
     - ``KOKKOS_VERSION`` and the comparison helpers, to check which Kokkos version is in use.

Configuration macros
--------------------

These macros let you introspect how Kokkos was configured, for example which
backends are enabled. Unlike the macros above, they are defined only if the
corresponding option was enabled, so test them with ``#ifdef`` or
``#if defined(...)``.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 70

   * - Page
     - Content
   * - :doc:`General settings <macros/general>`
     - Debugging, deprecation, tuning, C++ standard and third-party libraries.
   * - :doc:`Backends <macros/backends>`
     - Which execution spaces are enabled, and backend-specific options.
   * - :doc:`Architectures <macros/architectures>`
     - ``KOKKOS_ARCH_*``: the hardware Kokkos was configured for. Your code should not depend on these.

.. _macros_configuration_relationship:

Relationship with CMake configuration options
---------------------------------------------

The ``KOKKOS_ENABLE_*`` and ``KOKKOS_ARCH_*`` macros are, for the most part, the counterpart of a
CMake option documented in the :doc:`Configuration Guide
<../../get-started/configuration-guide>`.  The names follow a simple rule:

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 40 40

   * - CMake option
     - Macro
   * - ``Kokkos_ENABLE_<NAME>=ON``
     - ``KOKKOS_ENABLE_<NAME>`` is defined
   * - ``Kokkos_ARCH_<NAME>=ON``
     - ``KOKKOS_ARCH_<NAME>`` is defined
   * - ``CMAKE_CXX_STANDARD=<N>``
     - ``KOKKOS_ENABLE_CXX<N>`` is defined

These macros are either defined or not: test them with ``#if defined(...)``,
``#ifdef`` or ``#ifndef``, not with their value.  Options that are ``OFF`` leave the
macro undefined.  Exceptions to the naming rule, and macros that have no CMake
counterpart, are called out on each page.

.. code-block:: c++

   #include <Kokkos_Core.hpp>

   #ifdef KOKKOS_ENABLE_CUDA
   // Kokkos was configured with -DKokkos_ENABLE_CUDA=ON
   #endif

.. toctree::
   :hidden:
   :maxdepth: 1

   macros-special/host_device_macros
   macros-special/if_on_host_or_device
   macros/version
   macros/general
   macros/backends
   macros/architectures
