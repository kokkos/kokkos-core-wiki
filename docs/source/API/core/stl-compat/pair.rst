``Kokkos::pair``
================

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Pair.hpp>``

An implementation of ``std::pair`` that is intended to be fully compatible, with the exception that ``Kokkos::pair`` will work on the device. Also provides utility functions to convert from and to ``std::pair``

Usage
-----

.. code-block:: cpp

    std::pair<int, float> std_pair = std::make_pair(1,2.0f); 
    Kokkos::pair<int_float> kokkos_pair = Kokkos::make_pair(1,2.0f);
    Kokkos::pair<int, float> converted_std_pair(std_pair);
    std::pair<int,float> converted_kokkos_pair = kokkos_pair.to_std_pair();

Description
-----------

.. cppkokkos:struct:: template <class T1, class T2> pair

    |

    .. rubric:: Public Typedefs

    .. cppkokkos:type:: T1 first_type;

        The type of the first element in the pair.

    .. cppkokkos:type:: T2 second_type;

        The type of the second element in the pair.
    
    .. cppkokkos:var:: first_type first;

        The first element in the pair.

    .. cppkokkos:var:: second_type second;

        The second element in the pair.

    .. rubric:: Constructors

    .. cppkokkos:function:: KOKKOS_DEFAULTED_FUNCTION constexpr pair() = default;

        Default constructor. Initializes both data members with their defaults.

    .. cppkokkos:function:: KOKKOS_FORCEINLINE_FUNCTION constexpr pair(first_type const& f, second_type const& s);

        Element-wise constructor. Assigns ``first`` the value of ``f``, ``second`` the value of ``s``.

    .. cppkokkos:function:: template <class U, class V> KOKKOS_FORCEINLINE_FUNCTION constexpr pair(const pair<U, V>& p);

        Conversion from ``std::pair``. Assigns each element of the pair to its corresponding element in the ``p``.

    .. rubric:: Assignment and conversion

    .. cppkokkos:function:: template <class U, class V> KOKKOS_FORCEINLINE_FUNCTION pair<T1, T2>& operator=(const pair<U, V>& p);

        Sets ``first`` to ``p.first`` and ``second`` to ``p.second``.

    .. rubric:: Functions

    .. cppkokkos:function:: std::pair<T1, T2> to_std_pair() const;

        Returns a ``std::pair`` whose contents match those of the ``Kokkos::pair``. Useful for interacting with libraries that explicitly only accept ``std::pair``.
