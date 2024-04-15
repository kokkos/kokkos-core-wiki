Known issues
############

.. role:: cppkokkos(code)
    :language: cppkokkos


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
  for comparators used with Kokkos::sort in Kokkos versions prior to 4.3. The best advice to give is to make sure the respective
  parameters are trivially-copyable. If this isn't possible, sycl::is_device_copyable should be specialized and users should make
  sure to use raw pointers instead of Kokkos::Views.

  .. code-block:: cpp

     Kokkos::sort(exec, values, KOKKOS_LAMBDA(int i, int j) { return keys(i) < keys(j); });

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

Mathematical functions
======================

- Compatibilty issue with using-directives and mathematical functions:

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
