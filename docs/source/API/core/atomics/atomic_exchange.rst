``atomic_exchange``
===================

.. role:: cppkokkos(code)
   :language: cppkokkos

Header File: <Kokkos_Core.hpp>

Usage
-----

.. code-block:: cpp

   old_val = atomic_exchange(ptr_to_value, new_value);

Atomically sets the value at the address given by ``ptr_to_value`` to ``new_value`` and returns the previously stored value at the address.

Description
-----------

.. cppkokkos:function:: template<class T> T atomic_exchange(T* const ptr_to_value, const T new_value);

   Atomically executes ``old_value = *ptr_to_value; *ptr_to_value = new_value; return old_value;``,
   where ``old_value`` is the value at address ``ptr_to_value`` before doing the exchange.

   :param ptr_to_value: address of the value to be updated

   :param new_value: new value
