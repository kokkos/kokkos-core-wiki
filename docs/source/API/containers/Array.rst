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

``Array`` is a contiguous aggregate owning container storing a fixed size sequence of objects (models holding exactly N elements).

* This container is an owning container (the data is embeddded in the container itself).
* This container is an aggregate type with the same semantics as a struct holding a C-style array ``T[N]`` as its only non-static data member when ``N > 0``; otherwise, it is an empty container.
* Unlike a C-style array, it doesn't decay to ``T*`` automatically.
* As an aggregate type, it can be initialized with aggregate-initialization given at most ``N`` initializers that are convertible to ``T``: ``Kokkos::Array<int, 3> a = { 1, 2, 3 };``.

..
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

    :return: A pointer to the first element of the array.  If ``N == 0``, the return value is unspecified and not dereferenceable.


Deduction Guides
----------------

.. cppkokkos:function:: template<class T, class... U> Array(T, U...) -> Array<T, 1 + sizeof...(U)>

Non-Member Functions
--------------------

..
  These should only be listed here if they are closely related. E.g. friend operators. However,
  something like view_alloc shouldn't be here for view

.. cppkokkos:function:: template<class T, size_t N> constexpr bool operator==(const Array<T, N>& l, const Array<T, N>& r) noexcept

   :return: ``true`` if and only if ∀ the elements in ``l`` and ``r`` compare equal.

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


Deprecated since 4.4.00:
------------------------
.. cppkokkos:struct:: template<class T = void, size_t N = KOKKOS_INVALID_INDEX, class Proxy = void> Array

* The primary template was an contiguous aggregate owning container of exactly ``N`` elements of type ``T``.
* This container did not support move semantics.

.. cppkokkos:struct:: template<class T, class Proxy> Array<T, 0, Proxy>

* This container was an empty container.

.. cppkokkos:struct:: template<class T> Array<T, KOKKOS_INVALID_INDEX, Array<>::contiguous>

* This container was a non-owning container.
* This container had its size determined at construction time.
* This container could be assigned from any ``Array<T, N , Proxy>``.
* Assignment did not change the size of this container.
* This container did not support move semantics.

.. cppkokkos:struct:: template<class T> Array<T, KOKKOS_INVALID_INDEX, Array<>::strided>

* This container was a non-owning container.
* This container had its size and stride determined at construction time.
* This container could be assigned from any ``Array<T, N , Proxy>``.
* Assignment did not change the size or stride of this container.
* This container did not support move semantics.

.. cppkokkos:struct:: template<> Array<void, KOKKOS_INVALID_INDEX, void>

   .. rubric:: Public Types

   .. cppkokkos:type:: contiguous
   .. cppkokkos:type:: stided

* This specialization defined the embedded tag types: ``contiguous`` and ``strided``.

