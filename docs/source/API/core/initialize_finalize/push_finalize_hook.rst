``push_finalize_hook``
======================

.. role::cpp(code)
    :language: cpp

Defined in header ``<Kokkos_InitializeFinalize.hpp>``:sup:`since Kokkos 5.3` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::push_finalize_hook(func);

Registers the callable object ``func`` to be called when the Kokkos execution
environment is terminated.

The functions registered via ``push_finalize_hook()`` will be called in reverse
order (last-in, first-out) when entering :cpp:func:`finalize`, before releasing
acquired resources and finalizing all backends.

If a function exits via a thrown exception, ``std::terminate`` is called.

Interface
---------

.. cpp:function:: void push_finalize_hook(std::function<void()> func);

   :param func: Function object to be called when entering :cpp:func:`finalize`

   .. versionchanged:: 5.3

      Thread-safe: may be called concurrently from multiple threads without
      additional synchronization.


Notes
-----
.. note::

   :cpp:func:`push_finalize_hook` may be called at any point in the program,
   including before :cpp:func:`initialize`. Hooks registered before
   :cpp:func:`initialize` are retained and invoked during :cpp:func:`finalize`,
   just like hooks registered after initialization. Because hooks run in reverse
   order of registration, a hook registered before :cpp:func:`initialize` will
   be among the *last* to execute.

   Conversely, since :cpp:func:`finalize` may only be called once per program,
   any hook registered after :cpp:func:`finalize` has run will never be called.


Example
-------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <iostream>

    void my_hook() {
      std::cout << "Cruel world!\n";
    }

    int main(int argc, char* argv[]) {
        // Legal to register a hook before Kokkos::initialize()
        Kokkos::push_finalize_hook(my_hook);
        Kokkos::initialize(argc, argv);
        Kokkos::push_finalize_hook([]{ std::cout << "Goodbye\n"; });
        std::cout << "Calling Kokkos::finalize() ...\n";
        Kokkos::finalize();
        // Never called: finalize() has already run and may only be called once,
        // so this hook is never invoked (otherwise it would call std::terminate).
        Kokkos::push_finalize_hook([]{ throw 42; });
    }


Output:

.. code-block::

    Calling Kokkos::finalize() ...
    Goodbye
    Cruel world!


See also
--------

.. seealso::

  :doc:`finalize`
    Terminate the Kokkos execution environment
