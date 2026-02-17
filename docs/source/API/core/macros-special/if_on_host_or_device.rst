.. _kokkos_if_on_host_device_macros:

=================================================
``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE``
=================================================

.. role:: cpp(code)
   :language: cpp

Overview
========
The ``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE`` macros are a pair of
function-like macros introduced in Kokkos 3.6 that enable **portable
conditional compilation** within a **single ``KOKKOS_FUNCTION`` body**. They
allow you to select which code is compiled and executed based on whether the
function is running on the host (CPU) or a device (GPU, etc.). These macros
provide an alternative to non-portable preprocessor idioms like #ifdef
``__CUDA_ARCH__``.

Motivation
==========
Traditional preprocessor directives like ``#ifdef __CUDA_ARCH__`` rely on a
split compilation model, where host and device code are compiled in separate
passes.  While this model is supported by some compilers (like ``nvcc``), it is
not universally portable.  Other modern compilers for GPU-accelerated code,
such as those that support OpenACC or OpenMPTarget, use a unified compilation
approach where both host and device code are compiled in a single pass. As a
result, code written with backend-specific macros is not portable across
different compilers and programming models

The ``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE`` macros solve this
portability problem by allowing the compiler to conditionally compile code
within a single pass, enabling a single code base to be used with a wider range
of compilers and backends.

Usage
=====
These macros are designed to be used within a function decorated with
``KOKKOS_FUNCTION``. They accept a single argument, which is a block of code
enclosed in double parentheses. The code inside the macro's parentheses will
only be compiled and executed on the specified architecture


Signature
---------

.. code-block:: cpp

    KOKKOS_IF_ON_HOST(( /* code to be compiled on the host */ ))
    KOKKOS_IF_ON_DEVICE(( /* code to be compiled on the device */ ))


Example: Host/Device Overloading
--------------------------------

A common use case is to provide different implementations of a function for
host and device execution.

.. code-block:: cpp

    struct MyS { int i; };

    KOKKOS_FUNCTION MyS MakeStruct() {
      // This return statement is only compiled for the host target.
      KOKKOS_IF_ON_HOST((
        return MyS{0};
      ))

      // This return statement is only compiled for the device target.
      KOKKOS_IF_ON_DEVICE((
        return MyS{1};
      ))
    }

Important Considerations
========================

Scope
-----

Each ``KOKKOS_IF_ON_*`` macro introduces a new scope. Any variables declared
within the macro's parentheses are local to that scope and will not be
accessible outside of it.

.. code-block:: cpp

    KOKKOS_IF_ON_HOST((
      int x = 0; // 'x' is only visible within this scope
      std::cout << x << '\n';
    )) // The scope of 'x' ends here.

``constexpr`` Context
---------------------

These macros **must not** be used in a context that requires a ``constexpr``
(constant expression). Using ``KOKKOS_IF_ON_HOST`` or ``KOKKOS_IF_ON_DEVICE``
within ``constexpr`` functions or to initialize ``constexpr`` variables leads to
**One Definition Rule (ODR) violations** and undefined behavior.

Why This Is Problematic
^^^^^^^^^^^^^^^^^^^^^^^^

Unlike runtime function calls, ``constexpr`` functions and variables generate
compile-time values that can affect the structure of types, the size of objects,
and template instantiations. When ``KOKKOS_IF_ON_HOST`` and
``KOKKOS_IF_ON_DEVICE`` are used in ``constexpr`` contexts, they cause the same
function or variable to have different compile-time values on the host versus
the device—similar to using architecture-specific preprocessor macros like
``#ifdef __AVX2__`` in different translation units.

This is analogous to the following problematic pattern:

.. code-block:: cpp

    // DO NOT DO THIS - ODR violation
    static constexpr int foo() {
      #ifdef __AVX2__
        return 4;
      #else
        return 2;
      #endif
    }

If you compile this code in two different translation units with different
compiler flags (one with AVX2 and one without) and then link them together, you
have an ODR violation because the same function has different definitions.

The same principle applies to host/device compilation: a ``constexpr`` function
that returns different values on host and device violates the ODR.

Examples of ODR Violations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problematic: Using in** ``constexpr`` **function**

.. code-block:: cpp

    // DO NOT DO THIS - causes ODR violation
    static constexpr KOKKOS_FUNCTION int compute_block_size() {
      KOKKOS_IF_ON_HOST((return 4;))
      KOKKOS_IF_ON_DEVICE((return 2;))
    }

    struct Functor {
      int data[compute_block_size()]; // Size differs on host vs device!
      // This creates an ODR violation and undefined behavior
    };

In this example, the ``Functor`` struct would have different sizes on the host
and device, causing serious memory corruption issues when passed between them.

**Problematic: Lambda captures with** ``constexpr`` **usage**

.. code-block:: cpp

    // DO NOT DO THIS - causes ODR violation
    void foo() {
      int a = 0;
      double b = 1.0;
      auto lambda = KOKKOS_LAMBDA(int) {
        KOKKOS_IF_ON_HOST((printf("%i\n", a);))     // Captures 'a' (4 bytes)
        KOKKOS_IF_ON_DEVICE((printf("%lf\n", b);))  // Captures 'b' (8 bytes)
      };
      // Lambda has different size on host (4 bytes) vs device (8 bytes)
    }

The lambda object has different sizes on host and device because of the
different captures, violating the ODR.

Correct Alternatives
^^^^^^^^^^^^^^^^^^^^

**Alternative 1: Use non-**``constexpr`` **runtime function**

If the value doesn't need to be a compile-time constant, simply remove
``constexpr``:

.. code-block:: cpp

    // This is OK - runtime function
    static KOKKOS_FUNCTION int compute_block_size() {
      KOKKOS_IF_ON_HOST((return 4;))
      KOKKOS_IF_ON_DEVICE((return 2;))
    }

**Alternative 2: Move** ``KOKKOS_IF_ON_*`` **to calling context**

If you need compile-time constants, move the conditional compilation up one
level:

.. code-block:: cpp

    // This is OK - separate constexpr values in each branch
    KOKKOS_INLINE_FUNCTION void process() {
      KOKKOS_IF_ON_HOST((
        constexpr int block_size = 4;
        // Use block_size here...
      ))
      KOKKOS_IF_ON_DEVICE((
        constexpr int block_size = 2;
        // Use block_size here...
      ))
    }

**Alternative 3: Use template specialization**

For more complex cases, consider using template specialization on execution
spaces:

.. code-block:: cpp

    // This is OK - different specializations
    template<typename ExecutionSpace>
    struct BlockSize {
      static constexpr int value = 2; // Default for devices
    };

    template<>
    struct BlockSize<Kokkos::DefaultHostExecutionSpace> {
      static constexpr int value = 4; // Specialized for host
    };

Best Practices
--------------

**Avoid using these macros whenever possible.**

``KOKKOS_IF_ON_HOST`` and ``KOKKOS_IF_ON_DEVICE`` should be considered a **last
resort** for code differentiation. The primary goal of Kokkos is to achieve
high-performance portability through a unified code base. Relying on these
macros can hinder this goal by introducing host/device-specific logic.

Before using these macros, consider alternative approaches like **partial
template specialization** on execution spaces or using Kokkos's built-in
functionalities, which are designed to be portable across all backends. Using
these macros should be limited to situations where a fundamental difference
between host and device APIs necessitates separate code paths, such as for I/O
operations or specific backend intrinsics.
