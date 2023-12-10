``atomic_compare_exchange_strong``
==================================

.. role:: cppkokkos(code)
   :language: cppkokkos

Header File: <Kokkos_Core.hpp>

Usage
-----

.. code-block:: cpp

   bool was_exchanged = atomic_compare_exchange_strong(ptr_to_value,
						       comparison_value,
						       new_value);

Atomically sets the value at the address given by ``ptr_to_value`` to ``new_value`` if the current value at ``ptr_to_value``
is equal to ``comparison_value``, and returns true if the exchange has happened.

Description
-----------

.. cppkokkos:function:: template<class T> bool atomic_compare_exchange_strong(T* const ptr_to_value, const T comparison_value, const T new_value);

   Atomically executes ``old_value = *ptr_to_value; if(old_value==comparison_value) *ptr_to_value = new_value; return old_value==comparison_value;``,
   where ``old_value`` is the value at address ``ptr_to_value`` before doing the exchange.

   :param ptr_to_value: address of the value to be updated

   :param comparison_value: value to be compared to

   :param new_value: new value
