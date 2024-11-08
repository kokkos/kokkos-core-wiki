``atomic_[op]``
===============

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    atomic_[op](ptr_to_value,update_value);

Atomically updates the ``value`` at the address given by ``ptr_to_value`` with ``update_value`` according to the relevant operation.

Description
-----------

.. cppkokkos:function:: template<class T> void atomic_add(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value += value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be added.

.. cppkokkos:function:: template<class T> void atomic_and(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value &= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.

.. cppkokkos:function:: template<class T> void atomic_decrement(T* const ptr_to_value);

   Atomically executes ``(*ptr_to_value)--`` or calls ``atomic_fetch_sub(ptr_to_value, T(-1))``.

   * ``ptr_to_value``: address of the to be updated value.

.. cppkokkos:function:: template<class T> void atomic_increment(T* const ptr_to_value);

   Atomically executes ``(*ptr_to_value)++`` or calls ``atomic_fetch_add(ptr_to_value, T(1))``.

   * ``ptr_to_value``: address of the to be updated value.

.. cppkokkos:function:: template<class T> void atomic_max(T* const ptr_to_value, const T value);

   Atomically executes ``if (value > *ptr_to_value) *ptr_to_value = value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value which to take the maximum with.

.. cppkokkos:function:: template<class T> void atomic_min(T* const ptr_to_value, const T value);

   Atomically executes ``if (value < *ptr_to_value) *ptr_to_value = value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value which to take the minimum with.

.. cppkokkos:function:: template<class T> void atomic_or(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value |= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value with which to combine the original value.

.. cppkokkos:function:: template<class T> void atomic_sub(T* const ptr_to_value, const T value);

   Atomically executes ``*ptr_to_value -= value``.

   * ``ptr_to_value``: address of the to be updated value.

   * ``value``: value to be subtracted.
