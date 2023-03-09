``atomic_compare_exchange``
===========================

.. role:: cppkokkos(code)
   :language: cppkokkos

Header File: <Kokkos_Core.hpp>

Usage:

.. code-block:: cpp

    old_val = atomic_compare_exchange(ptr_to_value,comparison_value, new_value);


Atomically sets the value at the address given by ``ptr_to_value`` to ``new_value`` if the current value at ``ptr_to_value``
is equal to ``comparison_value``, and returns the previously stored value at the address independent on whether
the exchange has happened.

Description
-----------

.. cppkokkos:function:: template<class T> T atomic_compare_exchange(T* const ptr_to_value, const T comparison_value, const T new_value);

   Atomically executes ``old_value = *ptr_to_value; if(old_value==comparison_value) *ptr_to_value = new_value; return old_value;``.

   :param ptr_to_value: address of the to be updated value

   :param comparison_value: value to be compared to

   :param new_value: new value

   :param old_value: value at address ``ptr_to_value`` before doing the exchange.
