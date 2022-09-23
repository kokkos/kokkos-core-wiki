``Kokkos::TestView``
====================

.. role:: cppkokkos(code)
   :language: cppkokkos

Header File: ``Kokkos_Core.hpp``

Class Interface
---------------

.. cppkokkos:class:: template <class DataType, class... Traits> TestView

  A potentially reference counted multi dimensional array with compile time layouts and memory space.
  Its semantics are similar to that of :cppkokkos:`std::shared_ptr`.

  .. rubric:: Public Member Variables

  .. cppkokkos:member:: static constexpr unsigned rank

    the rank of the view (i.e. the dimensionality).

  .. cppkokkos:type:: const_data_type

    The const version of ``DataType``, same as ``data_type`` if that is already const.

  .. cppkokkos:type:: uniform_runtime_type

    :cppkokkos:`uniform_type` but without compile time extents

  .. cppkokkos:deprecated-type:: 3.6.01 some_deprecated_data_type

    The const version of some_deprecated_data_type.

  .. cppkokkos:deprecated-type:: other_deprecated_data_type

    The const version of other_deprecated_data_type.

  .. cppkokkos:deprecated-type:: 3.7.0 some_deprecated_data_type_1

    The const version of some_deprecated_data_type_1.

  .. cppkokkos:function:: View( const ScratchSpace& space, const IntType& ... indices)

    *Requires:* :cppkokkos:`sizeof(IntType...)==rank_dynamic()` *and*  :cppkokkos:`array_layout::is_regular == true`.

    A constructor which acquires memory from a Scratch Memory handle.

    :param space: a scratch memory handle. Typically returned from :cppkokkos:`team_handles` in :cppkokkos:`TeamPolicy` kernels.
    :param indices: the runtime dimensions of the view.
