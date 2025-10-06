Known issues
############

.. role:: cpp(code)
    :language: cpp


Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE
=======================================

Building with the CMake language feature can cause problems in downstream libraries/applications.
CMake uses the file endings to determine the language a file should be compiled with. Since Kokkos files are named `cpp` and `hpp`, they are associated with `CXX` in CMake.
This implies that source and header files that use Kokkos might need to be redefined to be treated as another language. Otherwise, the language is detected based on the file endings. This might lead to files not being able to compile (e.g. using Kokkos in a `.cpp` file instead of a `.cu` file leads to CMake detecting `CXX` instead of `CUDA`).
Without specifying the language the compilation might fail depending on the capabilities of the `CXX` compiler to compile device code.
Furthermore, the architecture needs to be specified for every target in accordance with what `Kokkos_ARCH_...` is set and not with `CMAKE_<LANG>_ARCHITECTURES`. This also implies only one architecture can be active.

An example for marking the files accordingly can be found in `example/build_cmake_installed_kk_as_language`

Kokkos_ENABLE_MULTIPLE_CMAKE_LANGUAGES
======================================

This option allows to use one installed Kokkos library in multiple CMake languages (`CXX` and the language of the respective backend (`CUDA` or `HIP`)).
With this option enabled, Kokkos will use its compiler launcher script to redirect the `CXX` compiler unless the `separable_compilation` component is requested.
With the `separable_compilation` component, targets/projects/directories that link to Kokkos need to be marked manually via the CMake function `kokkos_compilation`.
Since Kokkos is limited to a single architecture, the `CMAKE_<LANG>_ARCHITECTURES` must correspond to the architecture enabled in Kokkos.

An example for using Kokkos with multiple languages can be found in `example/build_cmake_installed_multilanguage`

``Windows.h`` header
====================

When using Kokkos on Windows a program or library might include `windows.h`. This is problematic as this header defines two macros with the names `min` and `max` unless `NOMINMAX` is defined previously.
The preprocessor replaces strings in the source code with the macros yielding an uninterpretable result and thus compilation fails.
Therefore, the header `Kokkos_Core.hpp` is protected against these macros, meaning they are undefined at the beginning and redefined at the end of the header file.
Even though definitions inside `Kokkos_Core.hpp` are protected against the macros, code outside is not.
Thus, it is on the user to deal with the macros being defined, either by defining `-DNOMINMAX` or `/DNOMINMAX` in the compile line (preferred) or by putting `()` around names that contain `min` or `max`.

CUDA
====

- With some MPI versions or when using a legacy NVIDIA GPU, the default allocation mechanism of Kokkos (from version 4.2 to 4.4) for
  `CudaSpace` can cause issues. For example, MPI may crash with illegal memory accesses, or Kokkos' initialization
  can report errors like:

  .. code-block::

     terminate called after throwing an instance of 'Kokkos::Experimental::CudaRawMemoryAllocationFailure'

  A fix is to disable asynchronous memory allocations by adding the following to CMake arguments:

  .. code-block::

     -DKokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF

  A technical explanation of why disabling this policy is helpful for making some MPI implementations work, especially in their low-level layers
  like UCX, is partly due to the fact that `cudaMallocAsync` uses `cudaMemPool_t,` and the default memory pool
  does not support interprocess communication (IPC) without tweaking (https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/#interprocess_communication_support).
  The user should set up the default memory pool to properly support IPC (https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/#library_composability).

  Therefore, from version 4.5, the default behavior for Kokkos is to preventively disable `cudaMallocAsync.`

- CUDA 11.0 through 11.2 are not compatible with glibc 2.34 librt stubs. That issue is related to how the CMake package handles linking with librt. For more information please look at issue `#7512 <https://github.com/kokkos/kokkos/issues/7512>`_.

- Building an application that uses Kokkos with Microsoft Visual Studio and the `Cuda` backend enabled, requires the use of the CMake language feature, see :ref:`keywords_enable_backend_specific_options`.

HIP
===

- When using `HIPManagedSpace`, the memory migrates between the CPU and the GPU if:
   - the hardware supports it
   - the kernel was compiled to support page migration
   - the environment variable `HSA_XNACK` is set to 1

   See `here <https://docs.olcf.ornl.gov/systems/frontier_user_guide.html#enabling-gpu-page-migration>`_ for more explanation.

- Compatibility issue between HIP and gcc 8. You may encounter the following error:

  .. code-block::

     error: reference to __host__ function 'operator new' in __host__ __device__ function

  gcc 7, 9, and later do not have this issue.

SYCL
====

- Several of the Kokkos algorithm functions use third-party libraries like oneDPL.
  When using these, Kokkos doesn't control the kernel launch and thus the user has to make sure that all arguments
  that are forwarded to the TPL satisfy the sycl::is_device_copyable trait to avoid compiler errors. This holds true in particular
  for comparators used with Kokkos::sort in Kokkos versions prior to 4.7 and oneDPL versions prior to 2022.8.0.
  The best advice to give is to make sure the respective parameters are trivially-copyable, e.g., by using raw pointers instead of Kokkos::Views.
  If that's not possible and the oneDPL version is at least 2022.8.0, specializing sycl::is_device_copyable provides another workaround.

  .. code-block:: cpp

     MyComparator my_comparator;
     Kokkos::sort(exec, values, my_comparator);

  would give errors similar to

  .. code-block:: console

     /usr/bin/compiler/../../include/sycl/types.hpp:2572:17: error: static assertion failed due to requirement 'is_device_copyable_v<(lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1816:20)> || detail::IsDeprecatedDeviceCopyable<(lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1816:20), void>::value': The specified type is not device copyable
      2572 |   static_assert(is_device_copyable_v<FieldT> ||
           |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      2573 |                     detail::IsDeprecatedDeviceCopyable<FieldT>::value,
           |                     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      /usr/bin/compiler/../../include/sycl/types.hpp:2605:7: note: in instantiation of template class 'sycl::detail::CheckFieldsAreDeviceCopyable<(lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1578:83), 4>' requested here
      2605 |     : CheckFieldsAreDeviceCopyable<FuncT, __builtin_num_fields(FuncT)>,
           |       ^
     /usr/bin/compiler/../../include/sycl/types.hpp:2613:7: note: in instantiation of template class 'sycl::detail::CheckDeviceCopyable<(lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1578:83)>' requested here
      2613 |     : CheckDeviceCopyable<KernelType> {};
           |       ^
     /usr/bin/compiler/../../include/sycl/handler.hpp:1652:5: note: in instantiation of template class 'sycl::detail::CheckDeviceCopyable<sycl::detail::RoundedRangeKernel<sycl::item<1, true>, 1, (lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1578:83)>>' requested here
      1652 |     detail::CheckDeviceCopyable<KernelType>();
           |     ^
     /usr/bin/compiler/../../include/sycl/handler.hpp:1694:5: note: in instantiation of function template specialization 'sycl::handler::unpack<sycl::detail::RoundedRangeKernel<sycl::item<1, true>, 1, (lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1578:83)>, sycl::detail::RoundedRangeKernel<sycl::item<1, true>, 1, (lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1578:83)>, sycl::ext::oneapi::experimental::properties<std::tuple<>>, false, (lambda at /usr/bin/compiler/../../include/sycl/handler.hpp:1697:21)>' requested here
      1694 |     unpack<KernelName, KernelType, PropertiesT,
           |     ^
     /usr/bin/compiler/../../include/sycl/handler.hpp:1293:7: note: in instantiation of function template specialization 'sycl::handler::kernel_parallel_for_wrapper<sycl::detail::RoundedRangeKernel<sycl::item<1, true>, 1, (lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1578:83)>, sycl::item<1, true>, sycl::detail::RoundedRangeKernel<sycl::item<1, true>, 1, (lambda at /usr/include/oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h:1578:83)>, sycl::ext::oneapi::experimental::properties<std::tuple<>>>' requested here
      1293 |       kernel_parallel_for_wrapper<KName, TransformedArgType, decltype(Wrapper),
           |       ^
     /usr/bin/compiler/../../include/sycl/handler.hpp:2332:5: note: (skipping 7 contexts in backtrace; use -ftemplate-backtrace-limit=0 to see all)
      2332 |     parallel_for_lambda_impl<KernelName, KernelType, 1, PropertiesT>(
           |     ^
     [...]

  this is fixed by

  .. code-block:: cpp

    struct sycl::is_device_copyable<MyComparator>
      : std::true_type {};


Mathematical functions
======================

- Compatibility issue with using-directives and mathematical functions:

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    
    using namespace Kokkos;  // avoid using-directives

    KOKKOS_FUNCTION void do_math() {
      auto sqrt5 = sqrt(5);  // error: ambiguous ::sqrt or Kokkos::sqrt?
    }


.. _Compatibility: ./ProgrammingGuide/Compatibility.html

.. |Compatibility| replace:: Kokkos compatibility guidelines

The using-directive ``using namespace Kokkos;`` is highly discouraged (see
|Compatibility|_) and will cause compilation errors in presence of unqualified
calls to mathematical functions.  Instead, prefer explicit qualification
``Kokkos::sqrt`` or an using-declaration ``using Kokkos::sqrt;`` at local
scope.

Mathematical constants
======================

- Avoid taking the address of mathematical constants in device code.  It is not supported by some toolchains, hence not portable.

.. code-block:: cpp

    #include <Kokkos_Core.hpp>

    KOKKOS_FUNCTION void do_math() {
      // complex constructor takes scalar arguments by reference!
      Kokkos::complex z1(Kokkos::numbers::pi);
      // error: identifier "Kokkos::numbers::pi" is undefined in device code

      // 1*pi is a temporary
      Kokkos::complex z2(1 * Kokkos::numbers::pi);  // OK

      // copy into a local variable
      auto pi = Kokkos::numbers::pi;
      Kokkos::complex z3(pi);  // OK
    }
