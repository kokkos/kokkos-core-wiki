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

----

To ensure compliance with these standards and reduce CI noise, Kokkos utilizes
`pre-commit <https://pre-commit.com>`__ to automate linting and formatting.
This tool runs a series of "hooks" on your staged changes to ensure they meet
our standards for C++ (``clang-format``), CMake (``cmake-format``), and
metadata.

Environment Setup
^^^^^^^^^^^^^^^^^
To avoid conflicts with system-level packages, we recommend installing
``pre-commit`` within a Python virtual environment:

.. code-block:: bash

   # Create and activate a virtual environment
   python3 -m venv .kokkos-venv
   source .kokkos-venv/bin/activate

   # Install pre-commit
   pip install pre-commit

Installation and Automated Usage (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To have these checks run automatically every time you execute ``git commit``,
install the git hook scripts:

.. code-block:: bash

   pre-commit install

Once installed, if a hook finds an issue, it will automatically apply the fix
and "fail" the commit. You can then restage the fixed files and commit again.

Manual Execution and Targeted Checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The first time you run ``pre-commit``, it will download and build the
environments for the formatting tools. This initial setup can take several
minutes, but subsequent runs are cached and fast.

If you prefer to run specific tools directly without waiting for the entire
suite, you can invoke them by their hook ID:

* **Run all checks on staged changes:**
  ``pre-commit run``
* **Run only Clang-format (C++ files):**
  ``pre-commit run clang-format``
* **Run only CMake-format:**
  ``pre-commit run cmake-format``
* **Run a specific check on all files in the repository:**
  ``pre-commit run clang-format --all-files``

Leveraging these hooks locally ensures that your contributions are clean before
they reach the CI, allowing reviewers to focus on technical logic rather than
formatting minutiae.

.. note::
   While you can bypass hooks using ``git commit --no-verify``, this is
   discouraged. The CI will still enforce these checks and will fail the build
   if the standards are not met.

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

Be consistent with the placement of specifiers and qualifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Both the ``const West`` style (specifier/qualifier on the left) and the
``East const`` style (specifier/qualifier on the right) are acceptable. We do
not impose one over the other. We only request that you don't mix the two
styles within the same block of code.

There is one exception: we request you please always use ``constexpr West``.

Additionally, when multiple qualifiers are present, they must appear on the
same side. Don't split them (e.g. ``const int volatile``).

Don't:

.. code-block:: cpp

    // Mixing left and right alignment in the same block
    const int i  = 42;
    auto const f = 3.14f;

    // constexpr on the right
    int constexpr c = 3;

    // Qualifiers split on both sides
    const int volatile d = 4;


Do:

.. code-block:: cpp

    // Consistent left alignment within the block
    const int i  = 42;
    const auto f = 3.14f;

    // Or consistent right alignment within the block
    int const i  = 42;
    auto const f = 3.14f;

    // constexpr always on the left
    constexpr int c = 3;

    // Qualifiers grouped on the same side
    const volatile int d = 4;
    // or
    int const volatile d = 4;

    // A const pointer to a const, using either style
    const int* const p   = &i;
    float const* const q = &f;

Symbol naming style conventions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following conventions are based on the most commonly used patterns in Kokkos Core.
These are guidelines, and they were not always consistently followed.

* **Classes and Concepts**: Use ``UpperCamelCase`` (for example ``View``,
  ``Device``, ``ExecutionSpace``, ``GraphNodeImpl``).
* **Template parameters**: Use semantic ``UpperCamelCase`` names for type
  parameters (for example ``ExecutionSpace``, ``MemorySpace``, ``DataType``,
  ``FunctorType``). Variadic packs typically use descriptive plural names like
  ``Properties`` or ``Args``. Short names (``T``, ``P``, etc.) are mostly used
  in local/internal contexts.
* **Macros**: Use all-caps with underscores and a ``KOKKOS_`` prefix. Use
  ``KOKKOS_IMPL_`` for internal-only macros.
* **Functions (including member functions)**: Use ``lower_snake_case`` names
  (for example ``parallel_for``, ``create_mirror_view_and_copy``,
  ``impl_static_fence``, ``print_configuration``). Non-public member functions that
  for implementation reasons can't be made ``private`` use an ``impl_`` prefix.
* **Class data members**: Use ``m_`` + ``lower_snake_case`` (for example
  ``m_space_instance``, ``m_thread_team_data``, ``m_queue``).
* **Namespaces**: ``Kokkos`` for public APIs, with ``Kokkos::Impl`` and
  ``Kokkos::Experimental`` used to scope internal or experimental symbols.
* **Type aliases and traits aliases**: Commonly use ``lower_snake_case`` with
  ``_type`` suffixes where helpful (for example ``execution_space``,
  ``value_type``, ``device_type``).

Use an anonymous namespace in unit tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Test files commonly declare helper functions, functors, and types at
namespace scope (fixtures, ``operator()`` structs for ``parallel_for``, free
functions, etc.). Since every ``.cpp`` test file is compiled into the same
test executable, any such symbol with external linkage can clash with an
identically named symbol defined in another test file. This is a genuine ODR
violation: at best it is a duplicate-symbol link error, and at worst the
linker silently picks one definition and a test ends up calling the wrong
one.

Wrapping the test file's contents in a namespace called ``Test`` does **not**
solve this. ``namespace Test { ... }`` is just a regular, open namespace:
nearly every test file in the suite reopens the exact same ``Test``
namespace, so a function or class with the same name defined in two different
test files still collides, exactly as if neither had used a namespace at
all. The namespace name does nothing to make the symbols unique because it
is shared, not per-file.

The robust fix is to put test-local declarations in an unnamed (anonymous)
namespace instead. Each translation unit gets its own, distinct anonymous
namespace, so every symbol declared inside one is given internal linkage and
cannot collide with a same-named symbol in any other ``.cpp`` file; there is
no need to invent a unique name per file or per test.

Don't:

.. code-block:: cpp

    #include <gtest/gtest.h>
    #include <Kokkos_Core.hpp>

    namespace Test {
    struct Functor {
      KOKKOS_FUNCTION void operator()(int i) const { /* ... */ }
    };

    void run_test() { /* ... */ }
    }  // namespace Test

    TEST(TEST_CATEGORY, functor) {
      Test::run_test();
    }

Do:

.. code-block:: cpp

    #include <gtest/gtest.h>
    #include <Kokkos_Core.hpp>

    namespace {
    struct Functor {
      KOKKOS_FUNCTION void operator()(int i) const { /* ... */ }
    };

    void run_test() { /* ... */ }
    }  // namespace

    TEST(TEST_CATEGORY, functor) {
      run_test();
    }
