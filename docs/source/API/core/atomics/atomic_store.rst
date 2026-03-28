``atomic_store``
================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    atomic_store(&obj, desired);

Atomically replaces the current value of ``obj`` with ``desired``.

Description
-----------

.. cpp:function:: template<class T> void atomic_store(T* ptr, std::type_identity_t<T> val);

   Atomically writes ``val`` into ``*ptr``.

   ``{ *ptr = val; }``

   :param ptr: address of the object whose value is to be replaced
   :param val: the value to store in the referenced object
   :returns: (nothing)


See also
--------
* :doc:`atomic_load <atomic_load>`: atomically obtains the value of the referenced object
* :doc:`atomic_exchange <atomic_exchange>`: atomically replaces the value of the referenced object and obtains the value held previously
