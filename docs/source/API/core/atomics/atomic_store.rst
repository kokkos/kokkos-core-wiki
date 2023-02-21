``atomic_store``
================

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    atomic_store(ptr_to_value,new_value);

Atomically sets the value at the address given by ``ptr_to_value`` to ``new_value``.

Synopsis
--------

.. cppkokkos:function:: template<class T> void atomic_store(T* const ptr_to_value, const T new_value);

Description
-----------

.. cppkokkos:function:: template<class T> void atomic_store(T* const ptr_to_value, const T new_value);

    * Atomically executes ``*ptr_to_value = new_value;``. 
        - ``ptr_to_value``: address of the to be updated value.
        - ``new_value``: new value.
