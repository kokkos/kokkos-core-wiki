``view_alloc()``
================

.. role:: cpp(code)
   :language: cpp

Header File: ``Kokkos_View.hpp``

Usage:

.. code-block:: cpp

     Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing, "ViewString");
     Kokkos::view_wrap(pointer_to_wrapping_memory);

Create View allocation parameter bundle from argument list. Valid argument list members are:

* label as :cpp:`C`-string or :cpp:`std::string`
* memory space instance of the :cpp:`View::memory_space` type
* execution space instance able to access :cpp:`View::memory_space`
* :cpp:`Kokkos::WithoutInitializing` to bypass initialization
* :cpp:`Kokkos::AllowPadding` to allow allocation to pad dimensions for memory alignment
* a pointer to create an unmanaged View wrapping that pointer

Synopsis
--------

.. cpp:function:: template <class... Args> \
                  view_alloc(Args const&... args)

.. cpp:function:: template <class... Args> \
                  view_wrap(Args const&... args)

Description
-----------

.. cpp:function:: template <class... Args> \
                  view_alloc(Args const&... args)

  Create View allocation parameter bundle from argument list.

  Restrictions:

  * ``args`` : Cannot contain a pointer to memory.

.. cpp:function:: template <class... Args> \
                  view_alloc(Args const&... args)

  Create View allocation parameter bundle from argument list.

  Restrictions:

  * ``args`` : Can only be a pointer to memory.
