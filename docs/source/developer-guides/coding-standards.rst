Kokkos Coding Standards
=======================

Source Code Formatting
~~~~~~~~~~~~~~~~~~~~~~

File Headers
^^^^^^^^^^^^
Every source file should have the Kokkos `SPDX <https://spdx.dev>`__ file
header with our license identifier and copyright notice.

The header block must appear at the very top of the file:

.. code-block:: cpp

  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  // SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

  // The rest of the file content follows here.


Header Guard
^^^^^^^^^^^^
The header file’s guard should reflect the all-caps file name and path, using
an underscore instead of path separator and extension marker.

The logic behind this convention is to ensure global uniqueness that extends
not only across the entire Kokkos repository, but also into the full Kokkos
ecosystem and within user applications that include Kokkos headers. This is
achieved through two key steps:

1.  Project Prefix: Starting all guards with ``PROJECT_NAME_`` to ensure
    the macro does not conflict with external libraries or system headers.
2.  Path and Name Derivation: Converting the full file path and name (e.g.,
    ``impl/Kokkos_GarbageCollector.hpp``) to uppercase, replacing path
    separators (``/``) and the extension marker (``.``) with underscores
    (``_``).

For example, the guard for ``impl/Kokkos_GarbageCollector.hpp`` should be
something like ``KOKKOS_IMPL_GARBAGE_COLLECTOR_HPP``.
Or the guard for a HIP backend-specific implementation file
``HIP/Kokkos_HIP_BorrowChecker.hpp`` should be
``KOKKOS_HIP_BORROW_CHECKER_HPP``.

Comment Formatting
^^^^^^^^^^^^^^^^^^
In general, prefer C++-style comments (``//`` for normal comments, ``///`` for
doxygen documentation comments).

Style Issues
~~~~~~~~~~~~

Don’t use ``inline`` when defining a function within the class definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
C++ implicitly treats any member function defined within the class body as
inline. Adding the ``inline`` keyword, or using ``KOKKOS_INLINE_FUNCTION``,
adds unnecessary syntactic noise without changing the compiler's behavior.

Don't:

.. code-block:: cpp

    class Foo {
    public:
      // Redundant: already implicitly inline
      inline void bar() { /* ... */ }

      // Redundant: KOKKOS_INLINE_FUNCTION expands to 'inline'
      KOKKOS_INLINE_FUNCTION void baz() { /* ... */ }
    };


Do:

.. code-block:: cpp

  class Foo {
  public:
    // Clean: standard C++ handles inlining
    void bar() { /* ... */ }

    // Correct: Provides __host__ __device__ tags; inlining is implicit
    KOKKOS_FUNCTION void baz() { /* ... */ }
  };
