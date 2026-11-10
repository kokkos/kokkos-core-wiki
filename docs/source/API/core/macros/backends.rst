Backends
========

``KOKKOS_ENABLE_<BACKEND>`` macros are defined when the corresponding backend
was enabled at configuration time, with ``-DKokkos_ENABLE_<BACKEND>=ON``.
This page lists them, along with the backend-specific memory spaces and options.

.. note::

   Application code should rarely need these macros.  Instead of naming a
   concrete execution space such as ``Kokkos::Cuda`` or ``Kokkos::OpenMP``, use
   ``Kokkos::DefaultExecutionSpace`` and ``Kokkos::DefaultHostExecutionSpace``.
   Likewise, prefer ``Kokkos::DefaultExecutionSpace::memory_space`` and
   ``Kokkos::HostSpace`` over backend-specific memory spaces.  This keeps your
   code portable across backends.

   The legitimate use of these macros is interoperability with the underlying
   programming model, as :ref:`shown below <macros_backends_interoperability>`.

Execution and memory spaces
---------------------------

Test these macros for existence, e.g. ``#ifdef KOKKOS_ENABLE_SERIAL``.  See
:ref:`keywords_backends` in the Configuration Guide for the restrictions on
combining backends.  Device backends also provide memory spaces of their own,
which only exist when the backend is enabled.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 25 45

   * - Macro
     - Execution space
     - Backend-specific memory spaces
   * - ``KOKKOS_ENABLE_SERIAL``
     - :cpp:class:`Serial`
     -
   * - ``KOKKOS_ENABLE_OPENMP``
     - :cpp:class:`OpenMP`
     -
   * - ``KOKKOS_ENABLE_THREADS``
     - :cpp:class:`Threads`
     -
   * - ``KOKKOS_ENABLE_CUDA``
     - :cpp:class:`Cuda`
     - :cpp:class:`CudaSpace`, :cpp:class:`CudaUVMSpace`, :cpp:class:`CudaHostPinnedSpace`
   * - ``KOKKOS_ENABLE_HIP``
     - :cpp:class:`HIP`
     - :cpp:class:`HIPSpace`, :cpp:class:`HIPManagedSpace`, :cpp:class:`HIPHostPinnedSpace`
   * - ``KOKKOS_ENABLE_HPX``
     - :cpp:class:`HPX`
     -
   * - ``KOKKOS_ENABLE_SYCL``
     - :cpp:class:`SYCL`
     - :cpp:class:`SYCLDeviceUSMSpace`, :cpp:class:`SYCLSharedUSMSpace`, :cpp:class:`SYCLHostUSMSpace`
   * - ``KOKKOS_ENABLE_OPENACC``
     - :cpp:class:`OpenACC`
     -

See :doc:`../execution_spaces` and :doc:`../memory_spaces` for details.

.. _macros_backends_interoperability:

Interoperability with the backend programming model
---------------------------------------------------

Guarding code with the backend macro lets you hand Kokkos resources to native
libraries or kernels, for example a CUDA stream:

.. code-block:: c++

   #include <Kokkos_Core.hpp>

   void do_work() {
   #ifdef KOKKOS_ENABLE_CUDA
     Kokkos::Cuda space;
     cudaStream_t stream = space.cuda_stream();
     // hand the stream to a CUDA library or a hand-written CUDA kernel
   #endif
   }

Backend-specific options
------------------------

These follow the same naming rule and are documented alongside their CMake
option under :ref:`keywords_enable_backend_specific_options` in the
Configuration Guide.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 45 55

   * - Macro
     - Description
   * - ``KOKKOS_ENABLE_CUDA_CONSTEXPR``
     - Relaxed constexpr functions are enabled for the CUDA backend.
   * - ``KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE``
     - Relocatable device code (RDC) is enabled for the CUDA backend.
   * - ``KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE``
     - Relocatable device code (RDC) is enabled for the HIP backend.
   * - ``KOKKOS_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE``
     - Relocatable device code (RDC) is enabled for the SYCL backend.
