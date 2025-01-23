..
  Use the following convention for headings:

    # with overline, for parts (collections of chapters)

    * with overline, for chapters

    = for sections

    - for subsections

    ^ for subsubsections

    " for paragraphs

..
  Class / method / container name)
  for free functions that are callable, preserve the naming convention, `view_alloc()`

``Array``
==============

.. role:: cppkokkos(code)
    :language: cppkokkos

..
  The (pulic header) file the user will include in their code

Header File: ``Kokkos_Core.hpp``

..
  High-level, human-language summary of what the thing does, and if possible, brief statement about why it exists (2 - 3 sentences, max);

Description
-----------

``Array`` is a contiguous aggregate container storing a fixed_size sequence of objects (models holding exactly N elements).

  The API of the entity.

Interface
---------

..
  The declaration or signature of the entity.

.. cppkokkos:struct:: template <class T, size_t N> Array

  ..
    Template parameters (if applicable)
    Omit template parameters that are just used for specialization/are deduced/ and/or should not be exposed to the user.

  .. rubric:: Template Parameters

  :tparam T: The type of the element being stored.
  :tparam N: The number of elements being stored.

  .. rubric:: Public Types

  .. cppkokkos:type:: value_type = T
  .. cppkokkos:type:: pointer = T*
  .. cppkokkos:type:: const_pointer = const T*
  .. cppkokkos:type:: reference = T&
  .. cppkokkos:type:: const_reference = const T&
  .. cppkokkos:type:: size_type = size_t
  .. cppkokkos:type:: difference_type = ptrdiff_t

  .. rubric:: Public Member Functions

  .. cppkokkos:function:: static constexpr bool empty()

    :return: ``N == 0``

  .. cppkokkos:function:: static constexpr size_type size()
  .. cppkokkos:function:: constexpr size_type max_size() const

    :return: ``N``

  .. cppkokkos:function:: template<class iType> constexpr reference operator[](const iType& i)
  .. cppkokkos:function:: template<class iType> constexpr const_reference operator[](const iType& i) const

    :tparam iType: An integral type or an unscoped enum type.

    :return: A reference to the ``i``-th element of the array.

  .. cppkokkos:function:: constexpr pointer data()
  .. cppkokkos:function:: constexpr const_pointer data() const

    :return: A pointer to the first element of the array.  If ``N == 0``, the return value is unspecified.

Non-Member Functions
--------------------

..
  These should only be listed here if they are closely related. E.g. friend operators. However,
  something like view_alloc shouldn't be here for view

.. cppkokkos:function:: template<class T, size_t N> constexpr bool operator==(const Array<T, N>& l, const Array<T, N>& r) noexcept

   :return: ``true`` if ∀ the elements in ``l`` and ``r`` ``l[i] == r[i]``.

.. cppkokkos:function:: template<class T, size_t N> constexpr bool operator!=(const Array<T, N>& l, const Array<T, N>& r) noexcept

   :return: ``!(l == r)``

.. cppkokkos:function:: template<class T, size_t N> constexpr kokkos_swap(Array<T, N>& l, Array<T, N>& r) noexcept(N == 0 || is_nothrow_swappable_V<T>)

   :return: If ``T`` is swappable or ``N == 0``, each of the elements in `l` and `r` are swapped via ``kokkos_swap``.

.. cppkokkos:function:: template<class T, size_t N> constexpr Array<remove_cv_t<T>, N> to_Array(T (&a)[N])
.. cppkokkos:function:: template<class T, size_t N> constexpr Array<remove_cv_t<T>, N> to_Array(T (&&a)[N])

   :return: An ``Array`` containing the elements copied/moved from ``a``.

.. cppkokkos:function:: template<size_t I, class T, size_t N> constexpr T& get(Array<T, N>& a) noexcept
.. cppkokkos:function:: template<size_t I, class T, size_t N> constexpr const T& get(const Array<T, N>& a) noexcept

   :return: ``a[I]`` for (tuple protocol / structured binding support)

.. cppkokkos:function:: template<size_t I, class T, size_t N> constexpr T&& get(Array<T, N>&& a) noexcept
.. cppkokkos:function:: template<size_t I, class T, size_t N> constexpr const T&& get(const Array<T, N>&& a) noexcept

   :return: ``std::move(a[I])`` (for tuple protocol / structured binding support)


