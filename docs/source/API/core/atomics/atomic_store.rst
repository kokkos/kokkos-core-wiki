``atomic_store``
================

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    atomic_store(&obj, desired);

Atomically replaces ``obj``'s value with ``desired``.

Description
-----------

.. cppkokkos:function:: template<class T> void atomic_store(T* ptr, std::identity_type_t<T> val);

   Atomically writes ``val`` into ``*ptr`` (``*ptr = val;``).

   - ``ptr``: address of the object whose value is to be replaced.

   - ``val``: value to store in the referenced object.


See also
--------
* `atomic_load <atomic_load.html>`_: atomically obtains the value of the referenced object
* `atomic_exchange <atomic_exchange.html>`_: atomically replaces the value of the referenced object and obtains the value held previously
