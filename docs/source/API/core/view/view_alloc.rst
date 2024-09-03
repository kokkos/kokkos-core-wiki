``view_alloc``
==============

.. role:: cppkokkos(code)
   :language: cppkokkos

Header File: ``<Kokkos_View.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing, "ViewString");
    Kokkos::view_wrap(pointer_to_wrapping_memory);

Create View allocation parameter bundle from argument list. Valid argument list members are:

* label as ``C``-string or ``std::string``

* memory space instance of the ``View::memory_space`` type

* execution space instance able to access ``View::memory_space``

* ``Kokkos::WithoutInitializing`` to bypass initialization

* ``Kokkos::AllowPadding`` to allow allocation to pad dimensions for memory alignment

* a pointer to create an unmanaged View wrapping that pointer


Description
-----------

.. cppkokkos:function:: template <class... Args> ALLOC_PROP view_alloc(Args const&... args)

   Create View allocation parameter bundle from argument list.

   ``args`` : Cannot contain a pointer to memory.

.. cppkokkos:function:: template <class... Args> ALLOC_PROP view_wrap(Args const&... args)

   Create View allocation parameter bundle from argument list.

   ``args`` : Can only be a pointer to memory.

.. cppkokkos:type:: ALLOC_PROP

   :cppkokkos:type:`ALLOC_PROP` is a special, unspellable implementation-defined type that is returned by :cppkokkos:func:`view_alloc`
   and :cppkokkos:func:`view_wrap`. It represents a bundle of allocator parameters, including the View label, the memory space instance,
   the execution space instance, whether to initialize the memory, whether to allow padding, and the raw pointer value (for wrapped unmanaged views).
