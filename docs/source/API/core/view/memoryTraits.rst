``MemoryTraits``
================

Header File: ``<Kokkos_MemoryTraits.hpp>``

:cpp:struct:`MemoryTraits` is the last template parameter of :cpp:class:`View`.

Struct Interface
----------------

.. cpp:struct:: template <unsigned N> MemoryTraits

  When provided to a multidimensional View, ``MemoryTraits`` allow passing extra information about the treatment of the allocation. The template argument is expected to be a bitwise OR of enumeration values described below.

.. rubric:: Nested type

.. cpp:type::  memory_traits

  A tag type signifying the memory access trait(s) denoted by ``N``.

.. rubric:: Member Variables

.. cpp:member::  static constexpr bool is_unmanaged

  A boolean that indicates whether the Unmanaged trait is enabled.

.. cpp:member::  static constexpr bool is_random_access

  A boolean that indicates whether the RandomAccess trait is enabled.

.. cpp:member::  static constexpr bool is_atomic

  A boolean that indicates whether the Atomic trait is enabled.

.. cpp:member::  static constexpr bool is_restrict

  A boolean that indicates whether the Restrict trait is enabled.

.. cpp:member::  static constexpr bool is_aligned

  A boolean that indicates whether the Aligned trait is enabled.

.. _ProgrammingGuide: ../../../ProgrammingGuide/View.html#memory-access-traits

.. |ProgrammingGuide| replace:: Programming Guide

Non-Member Enums
----------------

The following enumeration values are used to specify the memory access traits. Check the |ProgrammingGuide|_ for further information about how these traits can be used in practice.

.. cpp:enum:: MemoryTraitsFlags

  The following enumeration values are defined in this enumeration type.

.. cpp:enumerator:: Unmanaged

  This traits means that Kokkos does neither reference counting nor automatic deallocation for such Views. This trait can be associated with memory allocated in any memory space. For example, an *unmanaged view* can be created by wrapping raw pointers of allocated memory, while also specifying the execution or memory space accordingly.

.. cpp:enumerator:: RandomAccess

  Views that are going to be accessed irregularly (e.g., non-sequentially) can be declared as :cpp:enumerator:`RandomAccess`. 

.. cpp:enumerator:: Atomic

  In such a view, every access (read or write) to any element will be atomic. 

.. cpp:enumerator:: Restrict

  This trait indicates that the memory of this view doesn't alias/overlap with another data structure in the current scope. 

.. cpp:enumerator:: Aligned

  This trait provides additional information to the compiler that the memory allocation in this ``View`` has an alignment of 64. 

Non-Member Types
----------------

The following type aliases are also available in the ``Kokkos`` namespace.

.. cpp:type:: MemoryManaged = Kokkos::MemoryTraits<>;
.. cpp:type:: MemoryUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
.. cpp:type:: MemoryRandomAccess = Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>;

Examples
--------

.. code-block:: cpp

   Kokkos::View<DayaType, LayoutType, MemorySpace, Kokkos::MemoryTraits<SomeTrait | SomeOtherTrait> > my_view;

Example MemoryTraits type: ``Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>``

