``initialize``
==============

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Usage: 

.. code-block:: cpp

    Kokkos::initialize(argc, argv);
    Kokkos::initialize(Kokkos::InitializationSettings()  // (since 3.7)
                       .set_disable_warnings(true)
                       .set_num_threads(8)
                       .set_map_device_id_by("random"));

Initializes the Kokkos execution environment.
This function must be called before any other Kokkos API functions or
constructors.  There are a small number of exceptions, such as
``Kokkos::is_initialized`` and ``Kokkos::is_finalized``.
Kokkos can be initialized at most once; subsequent calls are erroneous.

The function has two overloads.
The first one takes the same two parameters as ``main()`` corresponding to
the command line arguments passed to the program from the environment in which
the program is run.  Kokkos parses the arguments for the flags that it
recognizes.  Whenever a Kokkos flag is seen, it is removed from ``argv``, and
``argc`` is decremented.
The second one takes a `Kokkos::InitializationSettings <InitializationSettings.html#kokkosInitializationSettings>`_ class object
which allows for programmatic control of arguments.
`Kokkos::InitializationSettings <InitializationSettings.html#kokkosInitializationSettings>`_ is implicitly constructible from the ``Kokkos::InitArguments``<sup>deprecated in version 3.7</sup>.

Interface
---------

.. code-block:: cpp

    Kokkos::initialize(int& argc, char* argv[]);                 //             (1)
    Kokkos::initialize(InitArguments const& arguments);          // (until 3.7) (2)
    Kokkos::initialize(InitializationSettings const& settings);  // (since 3.7) (3)
    
Parameters
~~~~~~~~~~

* ``argc``: Non-negative value, representing the number of command line
  arguments passed to the program.
* ``argv``: Pointer to the first element of an array of ``argc + 1`` pointers,
  of which the last one is null and the previous, if any, point to
  null-terminated multibyte strings that represent the arguments passed to the
  program.
* ``arguments``: (deprecated since version 3.7) C-style ``struct`` object is
  converted to ``Kokkos::InitializationSettings`` for backward compatibility.
* ``settings``: ``class`` object that contains settings to control the
  initialization of Kokkos.

Requirements
~~~~~~~~~~~~

  * ``Kokkos::finalize`` must be called after ``Kokkos::initialize``.
  * ``Kokkos::initialize`` generally should be called after ``MPI_Init`` when Kokkos is initialized within an MPI context.
  * User initiated Kokkos objects cannot be constructed until after ``Kokkos::initialize`` is called.
  * ``Kokkos::initialize`` may not be called after a call to ``Kokkos::finalize``.

Semantics
~~~~~~~~~

  * After calling ``Kokkos::initialize``, ``Kokkos::is_initialized()`` should return true.

Example
~~~~~~~

.. code-block:: cpp
    #include <Kokkos_Core.hpp>

    int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {  // scope to ensure that my_view destructor is called before Kokkos::finalize
        Kokkos::View<double*> my_view("my_view", 10);
    }  // scope of my_view ends here
    Kokkos::finalize();
    }    

See also
--------

* `Kokkos::InitializationSettings <InitializationSettings.html#kokkosInitializationSettings>`_
* `Kokkos::ScopeGuard <ScopeGuard.html#kokkosScopeGuard>`_