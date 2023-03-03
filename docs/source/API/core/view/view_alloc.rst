``view_alloc()``
================

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

.. cppkokkos:function:: template <class... Args> impl_defined view_alloc(Args const&... args)

   Create View allocation parameter bundle from argument list.

   ``args`` : Cannot contain a pointer to memory.

.. cppkokkos:function:: template <class... Args> impl_defined view_wrap(Args const&... args)

   Create View allocation parameter bundle from argument list.

   ``args`` : Can only be a pointer to memory.
