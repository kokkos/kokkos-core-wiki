``is_finalized``
==============

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Usage: 

.. code-block:: cpp

    Kokkos::is_finalized();


Queries the initialization status of Kokkos and returns ``true`` if Kokkos is finalized and ``false`` is Kokkos is not finalized. This function can be called prior or after Kokkos initialization or finalization.

Interface
---------

.. code-block:: cpp

    Kokkos::is_finalized();
    
Requirements
~~~~~~~~~~~~

* ``Kokkos::is_finalized`` can be called before ``Kokkos::initialize``.
* ``Kokkos::is_finalized`` can be called after ``Kokkos::finalize``.

Semantics
~~~~~~~~~

* After calling ``Kokkos::finalize()``, ``Kokkos::is_finalized()`` should return true.

Example
~~~~~~~

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

See also
--------

* `Kokkos::InitializationSettings <InitializationSettings.html#kokkosInitializationSettings>`_
* `Kokkos::ScopeGuard <ScopeGuard.html#kokkosScopeGuard>`_
