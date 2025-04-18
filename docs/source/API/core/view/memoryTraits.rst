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

When provided to a multidimensional View, ``MemoryTraits<N>`` enables better access of the underlying memory. The hints are provided as the non-type template parameter ``N`` (an unsigned int). ``N`` is obtained by bitwise OR of values from the enumeration type ``MemoryTraitsFlags`` described below.

Nested Typedefs
+++++++++++++++
   .. cppkokkos:type:: memory_traits

       A tag signifying that this models the Memory access trait(s) denoted by the value of N.

Member Variables
++++++++++++++++
   .. cppkokkos:member::  static constexpr unsigned impl_value

      An unsigned integer having the value of N.

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

.. code-block:: cpp

 enum MemoryTraitsFlags {
   Unmanaged    = 0x01,
   RandomAccess = 0x02,
   Atomic       = 0x04,
   Restrict     = 0x08,
   Aligned      = 0x10
 };

A few useful type aliases are also available in the ``Kokkos`` namespace.

.. code-block:: cpp

 using MemoryManaged   = Kokkos::MemoryTraits<0>;
 using MemoryUnmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
 using MemoryRandomAccess =
     Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>;

