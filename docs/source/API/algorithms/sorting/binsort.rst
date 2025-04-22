
.. role:: cppkokkos(code)
    :language: cppkokkos

``BinSort``
===========

Header File: ``<Kokkos_Sort.hpp>``.

This page describes the Kokkos' ``BinSort`` API.

API
^^^

BinSort Class
-------------

.. cpp:class:: template <class KeyViewType, class BinSortOp, class Space = typename KeyViewType::device_type, class SizeType = typename KeyViewType::memory_space::size_type> BinSort

   :tparam KeyViewType: View for the keys

   :tparam BinSortOp: TBD

   :tparam Space: TBD

   :tparam SizeType: TBD

   .. rubric:: Constructors

   .. cppkokkos:function:: template <typename ExecutionSpace> BinSort(const ExecutionSpace& exec, const_key_view_type keys, int range_begin, int range_end, BinSortOp bin_op, bool sort_within_bins = false)

      :param exec: execution space
      :param keys: dfkdfd
      :param range_begin: tbd
      :param range_end: tbd
      :param bin_op: dfdv
      :param sort_within_bins: dfdfd

   .. cppkokkos:function:: template <typename ExecutionSpace> BinSort(const ExecutionSpace& exec, const_key_view_type keys, BinSortOp bin_op, bool sort_within_bins = false)

      :param exec: execution space
      :param keys: dfkdfd
      :param bin_op: dfdv
      :param sort_within_bins: dfdfd

   .. cppkokkos:function:: BinSort(const_key_view_type keys, int range_begin, int range_end, BinSortOp bin_op, bool sort_within_bins = false)

      Constructor that takes the keys, the binning_operator and optionally whether to sort within bins (default false)

      :param keys: dfkdfd
      :param range_begin: tbd
      :param range_end: tbd
      :param bin_op: dfdv


   .. cppkokkos:function:: BinSort(const_key_view_type keys, BinSortOp bin_op, bool sort_within_bins = false)

      Another constructor

      :param keys: dfkdfd
      :param bin_op: dfdv
      :param sort_within_bins: dfdfd


   .. rubric:: Public Methods

   .. cppkokkos:function:: template <class ExecutionSpace> void create_permute_vector(const ExecutionSpace& exec)

      Create the permutation vector, the bin_offset array and the bin_count array.
      Can be called again if keys changed

   .. cppkokkos:function:: void create_permute_vector(const ExecutionSpace& exec)

      Create the permutation vector, the bin_offset array and the bin_count array.
      Can be called again if keys changed

   .. cppkokkos:function:: template <class ExecutionSpace, class ValuesViewType> void sort(const ExecutionSpace& exec, ValuesViewType const& values, int values_range_begin, int values_range_end) const

      Sort a subset of a view with respect to the first dimension using the
      permutation array


   .. cppkokkos:function:: template <class ValuesViewType> void sort(ValuesViewType const& values, int values_range_begin, int values_range_end) const

      Sort a subset of a view with respect to the first dimension using the permutation array


   .. cppkokkos:function:: template <class ExecutionSpace, class ValuesViewType> void sort(ExecutionSpace const& exec, ValuesViewType const& values) const

   .. cppkokkos:function:: template <class ValuesViewType> void sort(ValuesViewType const& values) const



|
|

Bin Op classes
--------------

.. cpp:class:: template <class KeyViewType> BinOp1D

   :tparam KeyViewType: View for the keys

   .. rubric:: Constructors

   .. cppkokkos:function:: BinOp1D(int max_bins, typename KeyViewType::const_value_type min, typename KeyViewType::const_value_type max)

      Construct BinOp with number of bins, minimum value and maximum value


.. cpp:class:: template <class KeyViewType> BinOp3D

   :tparam KeyViewType: View for the keys

   .. rubric:: Constructors

   .. cppkokkos:function:: BinOp3D(int max_bins[], typename KeyViewType::const_value_type min[], typename KeyViewType::const_value_type max[])

      Construct BinOp with number of bins, minimum values and maximum values

|
|
