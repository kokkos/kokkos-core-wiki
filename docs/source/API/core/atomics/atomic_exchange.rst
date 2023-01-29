``atomic_exchange``
===================

.. role::cpp(code)
    :language: cpp

Header File: ``Kokkos_Core.hpp``

Usage:

.. code-block:: cpp

    old_val = atomic_exchange(ptr_to_value,new_value);


Atomically sets the value at the address given by ``ptr_to_value`` to ``new_value`` and returns the previously stored value at the address.

Synopsis
--------

.. code-block:: cpp

    template<class T>
    T atomic_exchange(T* const ptr_to_value, const T new_value);

Description
-----------

- .. code-block:: cpp

    template<class T>
    T atomic_exchange(T* const ptr_to_value, const T new_value);

  Atomically executes ``old_value = *ptr_to_value; *ptr_to_value = new_value; return old_value;``. 

  - ``ptr_to_value``: address of the to be updated value.
  - ``new_value``: new value.
  - ``old_value``: value at address ``ptr_to_value`` before doing the exchange.