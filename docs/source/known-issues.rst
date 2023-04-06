Known issues
############

.. role:: cppkokkos(code)
    :language: cppkokkos


HIP
===

- Compatibility issue between HIP and gcc 8. You may encounter the following error:

  .. code-block::

     error: reference to __host__ function 'operator new' in __host__ __device__ function

  gcc 7, 9, and later do not have this issue.

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
|Compatibility|_) and will cause compileation errors in presence of unqualified
calls to mathematical functions.  Instead, prefer explicit qualification
``Kokkos::sqrt`` or an using-declaration ``using Kokkos::sqrt;`` at local
scope.
