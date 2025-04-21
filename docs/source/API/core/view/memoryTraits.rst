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

The following enumeration values are used to specify the memory access traits.

   - ``Kokkos::Unmanaged``
   - ``Kokkos::RandomAccess``
   - ``Kokkos::Atomic``
   - ``Kokkos::Restrict``
   - ``Kokkos::Aligned``

A few useful type aliases are also available in the ``Kokkos`` namespace.

.. code-block:: cpp

 using MemoryUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
 using MemoryRandomAccess =
     Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>;

