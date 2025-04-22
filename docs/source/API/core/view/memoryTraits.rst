``MemoryTraits``
================

.. role:: cppkokkos(code)
   :language: cppkokkos

Header File: ``<Kokkos_MemoryTraits.hpp>``

Usage
-----

``Kokkos::MemoryTraits`` is the last template parameter of Kokkos::View.

.. code-block:: cpp

   Kokkos::View<DayaType, LayoutType, MemorySpace, Kokkos::MemoryTraits<SomeTrait | SomeOtherTrait> > my_view;

Example MemoryTraits type: ``Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>``

Struct template
~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <unsigned N> struct MemoryTraits { ... };

Description
~~~~~~~~~~~

When provided to a multidimensional View, ``MemoryTraits`` allow passing extra information about the treatment of the allocation. The template argument is expected to be a bitwise OR of enumeration values described below.

Nested Typedefs
+++++++++++++++
   .. cppkokkos:type:: memory_traits

       A tag signifying that this models the Memory access trait(s) denoted by the value of N.

Member Variables
++++++++++++++++
   .. cppkokkos:member::  static constexpr bool is_unmanaged

       A boolean that indicates whether the Unmanaged trait is enabled.

   .. cppkokkos:member::  static constexpr bool is_random_access

       A boolean that indicates whether the RandomAccess trait is enabled.

   .. cppkokkos:member::  static constexpr bool is_atomic

       A boolean that indicates whether the Atomic trait is enabled.

   .. cppkokkos:member::  static constexpr bool is_restrict

       A boolean that indicates whether the Restrict trait is enabled.

   .. cppkokkos:member::  static constexpr bool is_aligned
 
       A boolean that indicates whether the Aligned trait is enabled.

.. _ProgrammingGuide: ../../../ProgrammingGuide/View.html#memory-access-traits

.. |ProgrammingGuide| replace:: Programming Guide

The following enumeration values are used to specify the memory access traits. Check the |ProgrammingGuide|_ for further information about how these traits can be used in practice.

   - ``Kokkos::Unmanaged``
     ``Unmanaged`` means that Kokkos does neither reference counting nor automatic deallocation for such Views. This trait can be associated with memory allocated in any memory space. For example, an *unmanaged view* can be created by wrapping raw pointers of allocated memory, while also specifying the execution or memory space accordingly.
   - ``Kokkos::RandomAccess``
     Views that are going to be accessed irregularly (e.g., non-sequentially) can be declared as `RandomAccess`. 
   - ``Kokkos::Atomic``
     In such a view, every access (read or write) to any element will be atomic. 
   - ``Kokkos::Restrict``
     The ``Restrict`` trait indicates that the memory of this view doesn't alias/overlap with another data structure in the current scope. 
   - ``Kokkos::Aligned``
     TBD

A few useful type aliases are also available in the ``Kokkos`` namespace.

.. code-block:: cpp

 using MemoryUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
 using MemoryRandomAccess = Kokkos::MemoryTraits<Kokkos::RandomAccess>;

