``initialization_settings``
===========================

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Usage:

.. code-block:: cpp

    auto settings = Kokkos::initialization_settings();


Returns the Kokkos initialization settings, which were resolved from the
argument(s) passed to ``Kokkos::initialized`` and the environment variables.
This function must be called after ``Kokkos::initialized`` and before
``Kokkos::finalized``.

Interface
---------

.. code-block:: cpp

    [[nodiscard]] Kokkos::initialization_settings() noexcept;  // (since 4.1)
    
Parameters
~~~~~~~~~~

* (none)
