``is_initialized``
==================

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Queries the initialization status of Kokkos and returns ``true`` if Kokkos is initialized and ``false`` if Kokkos is not initialized. This function can be called prior or after Kokkos initialization or finalization.

Interface
---------

.. code-block:: cpp

    Kokkos::is_initialized();
    
Requirements
~~~~~~~~~~~~

* ``Kokkos::is_initialized`` can be called before ``Kokkos::initialize``.
* ``Kokkos::is_initialized`` can be called after ``Kokkos::finalize``.

Semantics
~~~~~~~~~

* After calling ``Kokkos::initialize()``, ``Kokkos::is_initialized()`` should return true.

Example
~~~~~~~

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
