``is_finalized``
================

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Interface
---------

.. cpp:function:: [[nodiscard]] bool is_finalized() noexcept

   Queries the finalization status of Kokkos and returns ``true`` if Kokkos is finalized and ``false`` if Kokkos is not finalized. This function can be called prior or after Kokkos initialization or finalization.
   
   :return: ``true`` if :cpp:func:`finalize` has been called; `false` otherwise.

Examples
--------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <cassert>

    int main(int argc, char* argv[]) {
        assert(!Kokkos::is_finalized());
        Kokkos::initialize(argc, argv);
	assert(!Kokkos::is_finalized());
        Kokkos::finalize();
        assert(Kokkos::is_finalized());
    }    

.. seealso::

   `Kokkos::InitializationSettings <InitializationSettings.html#kokkosInitializationSettings>`_
      Define the settings for initializing Kokkos programmatically.
   `Kokkos::ScopeGuard <ScopeGuard.html#kokkosScopeGuard>`_
      A class to initialize and finalize Kokkos using RAII.
