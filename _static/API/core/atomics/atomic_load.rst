``atomic_load``
===============

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    auto current = atomic_load(&obj);

Atomically obtains ``obj``'s value.

Description
-----------

.. cppkokkos:function:: template<class T> T atomic_load(T* ptr);

   Atomically reads the content of ``*ptr`` and returns it (``T val = *ptr; return val;``). 

   - ``ptr``: address of the object whose current value is to be returned.

See also
--------
* `atomic_store <atomic_store.html>`_: atomically replaces the value of the referenced object with a non-atomic argument
* `atomic_exchange <atomic_exchange.html>`_: atomically replaces the value of the referenced object and obtains the value held previously
