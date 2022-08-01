Initialize and Finalize
=======================

Kokkos::initialize
------------------

Initializes Kokkos internal objects and all enabled Kokkos backends.

See `Kokkos::initialize <initialize_finalize/initialize.html>`_ for details.


Kokkos::finalize
----------------

Shutdown Kokkos initialized execution spaces and release internally managed resources.

See `Kokkos::finalize <initialize_finalize/finalize.html>`_ for details.


Kokkos::ScopeGuard
------------------

``Kokkos::ScopeGuard`` is a class which aggregates the resources managed by Kokkos. ScopeGuard will call ``Kokkos::initialize`` when constructed and ``Kokkos::finalize`` when destructed, thus the Kokkos context is automatically managed via the scope of the ScopeGuard object.

See `Kokkos::ScopeGuard <initialize_finalize/ScopeGuard.html>`_ for details.

ScopeGuard aids in the following common mistake which is allowing Kokkos objects to live past ``Kokkos::finalize``:

.. code-block:: cpp

  int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    Kokkos::View<double*> my_view("my_view", 10);
    Kokkos::finalize();
    // my_view destructor called after Kokkos::finalize !
  }

Switching to ``Kokkos::ScopeGuard`` fixes it:

.. code-block:: cpp

  int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Kokkos::View<double*> my_view("my_view", 10);
    // my_view destructor called before Kokkos::finalize
    // ScopeGuard destructor called, calls Kokkos::finalize
  }

In the above example, ``my_view`` will not go out of scope until the end of the main() function.  Without ``ScopeGuard``, ``Kokkos::finalize`` will be called before ``my_view`` is out of scope.  With ``ScopeGuard``, ``ScopeGuard`` will be dereferenced (subsequently calling ``Kokkos::finalize``) after ``my_view`` is dereferenced, which ensures the proper order during shutdown.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./initialize_finalize/initialize
   ./initialize_finalize/finalize
   ./initialize_finalize/ScopeGuard
   ./initialize_finalize/InitializationSettings
   ./initialize_finalize/InitArguments
