``atomic_assign``
=================

.. warning::
   Deprecated since Kokkos 4.5,
   use `atomic_store <atomic_store.html>`_ instead.

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Atomic.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    atomic_assign(&obj, desired);
    //     ^^^^^^
    // deprecated since Kokkos 4.5,
    // use atomic_store(&obj, desired) instead

Atomically replaces the current value of ``obj`` with ``desired``.

Description
-----------

.. cppkokkos:function:: template<class T> void atomic_assign(T* ptr, std::type_identity_t<T> val);

   Atomically writes ``val`` into ``*ptr``.

   ``{ *ptr = val; }``

   :param ptr: address of the object whose value is to be replaced
   :param val: the value to store in the referenced object
   :returns: (nothing)

