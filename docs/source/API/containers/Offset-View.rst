``OffsetView``
==============

Header File: ``Kokkos_OffsetView.hpp``

Description
-----------

An :cpp:class:`OffsetView` can be used when the indices of an array begin at something other than zero.
Its indices range from the begin index (inclusive) to the end index (exclusive) for each dimension.

.. warning::

  The OffsetView is in the Experimental namespace.

Interface
---------

.. cpp:class:: template <class DataType, class... Properties> OffsetView

  An :cpp:class:`OffsetView` is a :cpp:class:`View` whose indices begin at arbitrary values per dimension.
  It derives from :cpp:class:`View` and has the same template parameters.

  .. rubric:: Constructors:

  .. cpp:function:: OffsetView()

    The default constructor shares the properties of :cpp:class:`View`.
    Additionally, the begins are set to zero.

  .. cpp:function:: template <class RT, class... RP> OffsetView(const OffsetView<RT, RP...>& rhs)

    Copy constructor with a compatible :cpp:class:`OffsetView`.

    .. seealso:: :ref:`api-view-assignment`

  .. cpp:function:: template <class RT, class... RP> OffsetView(const View<RT, RP...>& aview, const begins_type& begins)

    Constructs an :cpp:class:`OffsetView` referencing the data of :cpp:any:`aview` with indices of dimension :cpp:`r` in the range :cpp:`[begin(r),begin(r) + extent(r))`.

    :param aview: the :cpp:class:`View` to reference.
    :param begins: the first index of each dimension.

  .. cpp:function:: template <class Begins, class Ends> explicit OffsetView(const std::string& name, const Begins& begins, const Ends& ends)

    Constructs an :cpp:class:`OffsetView` with indices of dimension :cpp:`r` in the range :cpp:`[begin(r),end(r))`.

    :tparam Begins: :cpp:`std::initializer_list<int64_t>`, :cpp:`Kokkos::Array`, :cpp:`std::array` or :cpp:`std::span` with a static extent.
    :tparam Ends: :cpp:`std::initializer_list<int64_t>`, :cpp:`Kokkos::Array`, :cpp:`std::array` or :cpp:`std::span` with a static extent.

    :param name: same as for :cpp:class:`View`.
    :param begins: the first index of each dimension (inclusive).
    :param ends: the end index of each dimension (exclusive).

    .. rubric:: Requirements:

    - :cpp:expr:`begins.size() == rank()` and :cpp:expr:`ends.size() == rank()`, which is checked at runtime for :cpp:`std::initializer_list`.
    - :cpp:expr:`begins[r] <= ends[r]` for all :cpp:`0 <= r < rank()`, which is checked at runtime.

  .. cpp:function:: template <class Begins, class Ends> explicit OffsetView(const ALLOC_PROP& prop, const Begins& begins, const Ends& ends)

    Allocating constructor with allocation properties (created by a call to :cpp:func:`view_alloc`).

    :tparam Begins: :cpp:`std::initializer_list<int64_t>`, :cpp:`Kokkos::Array`, :cpp:`std::array` or :cpp:`std::span` with a static extent.
    :tparam Ends: :cpp:`std::initializer_list<int64_t>`, :cpp:`Kokkos::Array`, :cpp:`std::array` or :cpp:`std::span` with a static extent.

    :param prop: same as for :cpp:class:`View`.
    :param begins: the first index of each dimension (inclusive).
    :param ends: the end index of each dimension (exclusive).

  .. cpp:function:: template <class Begins, class Ends> OffsetView(const pointer_type& ptr, const Begins& begins, const Ends& ends)

    Unmanaged data wrapping constructor.
    Indices of dimension :cpp:`r` are in the range :cpp:`[begin(r),end(r))`.

    :tparam Begins: :cpp:`std::initializer_list<int64_t>`, :cpp:`Kokkos::Array`, :cpp:`std::array` or :cpp:`std::span` with a static extent.
    :tparam Ends: :cpp:`std::initializer_list<int64_t>`, :cpp:`Kokkos::Array`, :cpp:`std::array` or :cpp:`std::span` with a static extent.

    :param ptr: pointer to a user provided memory allocation.
    :param begins: the first index of each dimension (inclusive).
    :param ends: the end index of each dimension (exclusive).

    .. rubric:: Requirements:

    - :cpp:expr:`begins.size() == rank()` and :cpp:expr:`ends.size() == rank()`, which is checked at runtime for :cpp:`std::initializer_list`.
    - :cpp:expr:`begins[r] <= ends[r]` for all :cpp:`0 <= r < rank()`, which is checked at runtime.

  .. rubric:: Public Member Functions:

  .. cpp:function:: template <class... OtherIndexTypes> constexpr reference_type operator()(OtherIndexTypes... indices) const

    :tparam OtherIndexTypes: types convertible to :cpp:type:`index_type`.

    :param indices: the indices of the element to get a reference to, starting at :cpp:func:`begin` of each dimension.
    :return: a reference to the element at the given indices.

  .. cpp:function:: template <class iType> int64_t begin(const iType dim) const

    :tparam iType: an integral type.

    :param dim: the dimension to get the first index of.
    :return: the first index of dimension :cpp:any:`dim`.

  .. cpp:function:: template <class iType> int64_t end(const iType dim) const

    :tparam iType: an integral type.

    :param dim: the dimension to get the end index of.
    :return: :cpp:expr:`begin(dim) + extent(dim)`, i.e., one more than the last index of dimension :cpp:any:`dim`.

  .. cpp:function:: view_type view() const

    :return: a :cpp:class:`View` with the properties of the corresponding :cpp:class:`OffsetView`.

Non-Member Functions
--------------------

.. cpp:function:: template <class LT, class... LP, class RT, class... RP> bool operator==(const OffsetView<LT, LP...>& lhs, const OffsetView<RT, RP...>& rhs)

  :return: :cpp:`true` if :cpp:`lhs.view() == rhs.view()` and :cpp:`lhs.begins() == rhs.begins()`.

.. cpp:function:: template <class LT, class... LP, class RT, class... RP> bool operator!=(const OffsetView<LT, LP...>& lhs, const OffsetView<RT, RP...>& rhs)

  :return: :cpp:expr:`!(lhs == rhs)`

.. cpp:function:: template <class D, class... P, class... Args> auto subview(const OffsetView<D, P...>& src, Args... args)

  Creates an :cpp:class:`OffsetView` referencing a subset of the data of :cpp:any:`src`.

  :param src: the :cpp:class:`OffsetView` to take the subview of.
  :param args: one slice per dimension of :cpp:any:`src`, given in the indices of :cpp:any:`src`.
    Accepted are an integral index, :cpp:`Kokkos::ALL()`, or a :cpp:`Kokkos::pair<T, T>` or :cpp:`std::pair<T, T>` with an integral :cpp:`T`.
  :return: an :cpp:class:`OffsetView`. A dimension sliced with :cpp:`Kokkos::ALL()` keeps its begin index, the indices of all other dimensions start at :cpp:`0`.

  .. seealso:: :doc:`../core/view/subview`


Examples
--------

.. code-block:: cpp

  #include <iostream>
  #include <Kokkos_Core.hpp>
  #include <Kokkos_OffsetView.hpp>

  int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
      std::cout << std::boolalpha;
      using Kokkos::Experimental::OffsetView;
      OffsetView<int64_t*> ov("ov", {-2}, {3});
      Kokkos::View<int64_t*> v = ov.view();

      Kokkos::parallel_for(
          "WriteIndices", Kokkos::RangePolicy(ov.begin(0), ov.end(0)),
          KOKKOS_LAMBDA(const int64_t i) { ov(i) = i; });

      auto ov_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ov);
      auto v_h  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
      std::cout << "ov_h(-1) == v_h(1): " << (ov_h(-1) == v_h(1)) << '\n';
      std::cout << "ov_h(-1): " << ov_h(-1) << '\n';
      std::cout << "v_h(1): " << v_h(1) << '\n';
      std::cout << '\n';

      // The begin index of a sub range is set to zero
      auto sub   = Kokkos::subview(ov, Kokkos::pair<int64_t, int64_t>{-1, 2});
      auto sub_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sub);
      std::cout << "sub_h(0) == ov_h(-1): " << (sub_h(0) == ov_h(-1)) << '\n';
      std::cout << "sub_h(0): " << sub_h(0) << '\n';
      std::cout << "ov_h(-1): " << ov_h(-1) << '\n';
      std::cout << '\n';

      // The begin index of a full range is kept
      auto sub_all = Kokkos::subview(ov, Kokkos::ALL());
      auto sub_all_h =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sub_all);
      std::cout << "sub_all_h(-2) == ov_h(-2): " << (sub_all_h(-2) == ov_h(-2))
                << '\n';
      std::cout << "sub_all_h(-2) = " << sub_all_h(-2) << '\n';
      std::cout << "ov_h(-2)      = " << ov_h(-2) << '\n';
    }
    Kokkos::finalize();
  }

