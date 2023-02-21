``atomic_load``
===============

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    value = atomic_load(ptr_to_value);

Atomically reads the value at the address given by ``ptr_to_value``.

Synopsis
--------

.. cppkokkos:function:: template<class T> T atomic_load(T* const ptr_to_value);

Description
-----------

.. cppkokkos:function:: template<class T> T atomic_load(T* const ptr_to_value);

    * Atomically executes ``value = *ptr_to_value; return value;``. 
        - ``ptr_to_value``: address of the to be updated value.
        - ``value``: value at address ``ptr_to_value``.
