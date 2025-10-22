``is_initialized``
==================

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Interface
---------

.. cpp:function:: [[nodiscard]] bool is_initialized() noexcept

   Queries the initialization status of Kokkos and returns ``true`` if Kokkos is initialized and ``false`` if Kokkos is not initialized. This function can be called prior or after Kokkos initialization or finalization.

   :return: ``true`` if :cpp:func:`initialize` has been called; `false` otherwise. 

Examples
--------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <cassert>

    int main(int argc, char* argv[]) {
        assert(!Kokkos::is_initialized());
        Kokkos::initialize(argc, argv);
	assert(Kokkos::is_initialized());
        Kokkos::finalize();
        assert(Kokkos::is_finalized());
    }    

See also
--------

* `Kokkos::InitializationSettings <InitializationSettings.html#kokkosInitializationSettings>`_
* `Kokkos::ScopeGuard <ScopeGuard.html#kokkosScopeGuard>`_
