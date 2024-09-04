
``Function markup macros``
==========================

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Macros.hpp>``

Usage: 

.. code-block:: cpp

    KOKKOS_FUNCTION void foo();
    KOKKOS_INLINE_FUNCTION void foo();
    KOKKOS_FORCEINLINE_FUNCTION void foo();
    auto l = KOKKOS_LAMBDA(int i) { ... };
    auto l = KOKKOS_CLASS_LAMBDA(int i) { ... }; 

These macros deal with the management of split compilation for device and host code.
They fullfill the same purpose as the ``__host__ __device__`` markup in CUDA and HIP.
Generally only functions marked with one of these macros can be used inside of parallel
Kokkos code - i.e. all code executed in parallel algorithms must be marked up by one
of these macros.

``KOKKOS_FUNCTION``
-------------------

This macro is the equivalent of ``__host__ __device__`` markup in CUDA and HIP.
Use it primarily on inline-defined member functions of classes and templated 
free functions

.. code-block:: cpp

    class Foo {
      public:
        // inline defined constructor
        KOKKOS_FUNCTION Foo() { ... };

        // inline defined member function
        template<class T>
        KOKKOS_FUNCTION void bar() const { ... }
    };

    template<class T>
    KOKKOS_FUNCTION void foo(T v) { ... }
         
This macro is also used for non-templated free functions in conjunction with relocatable device code -
i.e. if one wants to compile functions in some compilation unit A but call them from Kokkos
parallel constructs defined in compilation unit B.


``KOKKOS_INLINE_FUNCTION``
--------------------------

This macro is the equivalent of ``__host__ __device__ inline`` markup in CUDA and HIP.
Use it primarily for non-templated free functions:

.. code-block:: cpp

    KOKKOS_INLINE_FUNCTION void foo() {}

Note that it is NOT a bug to use this macro for inline-defined member function of classes, or
templated free functions. It is simply redundant since they are by default inline.

``KOKKOS_FORCEINLINE_FUNCTION``
-------------------------------

This macro is the equivalent of ``__host__ __device__`` markup in CUDA and HIP, but also uses
compiler dependent hints (if available) to enforce inlining.
This can help with some functions which are often used, but it may also hurt compilation time,
as well as runtime performance due to code-bloat. In some instances using ``KOKKOS_FORCEINLINE_FUNCTION``
excessively can even cause compilation errors due to compiler specific limits of maximum inline limits.
Use this macro only in conjunction with performing extensive performance checks.

.. code-block:: cpp

    class Foo {
      public:
        KOKKOS_FORCEINLINE_FUNCTION
        Foo() { ... };

        template<class T>
        KOKKOS_FORCEINLINE_FUNCTION
        void bar() const { ... }
    };

    template<class T>
    KOKKOS_FORCEINLINE_FUNCTION
    void foo(T v) { ... }
         
This macro is also used for non-templated free functions in conjunction with relocatable device code -
i.e. if one wants to compile functions in some compilation unit A but call them from Kokkos
parallel constructs defined in compilation unit B.


``KOKKOS_LAMBDA``
-----------------

This macro provides default capture clause and host device markup for lambdas. It is the equivalent of
``[=] __host__ __device__`` in CUDA and HIP.
It is used than creating C++ lambdas to be passed to Kokkos parallel dispatch mechanisms such as
``parallel_for``, ``parallel_reduce`` and ``parallel_scan``.

.. code-block:: cpp

    void foo(...) {
      ...
      parallel_for("Name", N, KOKKOS_LAMBDA(int i) {
        ...
      });
      ...
      parallel_reduce("Name", N, KOKKOS_LAMBDA(int i, double& v) {
        ...
      }, result);
      ...
    }

.. warning:: Do not use ``KOKKOS_LAMBDA`` inside functions marked as ``KOKKOS_FUNCTION`` etc. or within a lambda marked with ``KOKKOS_LAMBDA``. Specifically do not use ``KOKKOS_LAMBDA`` to define lambdas for nested parallel calls. CUDA does not support that. Use plain C++ syntax instead: ``[=] (int i) {...}``.

.. warning:: When creating lambdas inside of class member functions you may need to use ``KOKKOS_CLASS_LAMBDA`` instead.

``KOKKOS_CLASS_LAMBDA``
-----------------------

This macro provides default capture clause and host device markup for lambdas created inside of class member functions. It is the equivalent of
``[=, *this] __host__ __device__`` in CUDA and HIP, capturing the parent class by value instead of by reference.

.. code-block:: cpp

    class Foo {
      public:
        Foo() { ... };

        void bar() const {
          parallel_for("Name", N, KOKKOS_CLASS_LAMBDA(int i) {
            ...
          });
        }
    };

