``Kokkos::kokkos_swap``
=======================

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Swap.hpp>``:sup:`(since 4.3)` which is included from ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    template <class T>
    KOKKOS_FUNCTION constexpr void
    kokkos_swap(T& a, T& b) noexcept(std::is_nothrow_move_constructible_v<T> &&
                                     std::is_nothrow_move_assignable_v<T>);  // (1) (since 4.3)

    template <class T2, std::size_t N>
    KOKKOS_FUNCTION constexpr void
    kokkos_swap(T2 (&a)[N], T2 (&b)[N]) noexcept(noexcept(*a, *b));  // (2) (since 4.3)


1) Swaps the values ``a`` and ``b``. This overload does not participate in overload
   resolution unless ``std::is_move_constructible_v<T> && std::is_move_assignable_v<T>``
   is ``true``.

2) Swaps the arrays ``a`` and ``b``. This overload does not participate in
   overload resolution unless ``T2`` is swappable.
