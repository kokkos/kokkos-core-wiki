``ScatterView``
===============

.. role:: cppkokkos(code)
	:language: cppkokkos

Header File: ``<Kokkos_ScatterView.hpp>``

.. _parallelReduce: ../core/parallel-dispatch/parallel_reduce.html

.. |parallelReduce| replace:: :cpp:func:`parallel_reduce`

Description
-----------

.. cppkokkos:class:: template <typename DataType, int Op, typename ExecSpace, typename Layout, int contribution> ScatterView

    .. rubric:: Public Member Variables

    .. cppkokkos:type:: original_view_type

        Type of View passed to ScatterView constructor.

    .. cppkokkos:type:: original_value_type

        Value type of the original_view_type.

    .. cppkokkos:type:: original_reference_type

        Reference type of the original_view_type.

    .. cppkokkos:type:: data_type_info

        DuplicatedDataType, a newly created DataType that has a new runtime dimension which becomes the largest-stride dimension, from the given View DataType.

    .. cppkokkos:type:: internal_data_type

        Value type of data_type_info.

    .. cppkokkos:type:: internal_view_type

        A View type created from the internal_data_type.

    .. rubric:: Constructors

    .. cppkokkos:function:: ScatterView()

        The default constructor. Default constructs members.

    .. cppkokkos:function:: ScatterView(View<RT, RP...> const&)

        Constructor from a ``Kokkos::View``. ``internal_view`` member is copy constructed from this input view.

    .. cppkokkos:function:: ScatterView(std::string const& name, Dims ... dims)

        Constructor from variadic pack of dimension arguments. Constructs ``internal_view`` member.

    .. cppkokkos:function:: ScatterView(ALLOC_PROP const& arg_prop, Dims... dims)

        Constructor from variadic pack of dimension arguments. Constructs ``internal_view`` member.
        This constructor allows passing an object created by `Kokkos::view_alloc`` as first argument, e.g., for specifying an execution space via
        ``Kokkos::view_alloc(exec_space, "label")``.

    .. rubric:: Public Methods

    .. cppkokkos:function:: constexpr bool is_allocated() const

        :return: true if the ``internal_view`` points to a valid memory location. This function works for both managed and unmanaged views. With the unmanaged view, there is no guarantee that referenced address is valid, only that it is a non-null pointer.

    .. cppkokkos:function:: access() const

       use within a kernel to return a ``ScatterAccess`` member; this member accumulates a given thread's contribution to the reduction.

    .. cppkokkos:function:: subview() const

        :return: a subview of a ``ScatterView``

    .. cppkokkos:function:: contribute_into(View<DT, RP...> const& dest) const

       contribute ``ScatterView`` array's results into the input View ``dest``

    .. cppkokkos:function:: reset()

       performs reset on destination array

    .. cppkokkos:function:: reset_except(View<DT, RP...> const& view)

       tbd

    .. cppkokkos:function:: resize(const size_t n0 = 0, const size_t n1 = 0, const size_t n2 = 0, const size_t n3 = 0, const size_t n4 = 0, const size_t n5 = 0, const size_t n6 = 0, const size_t n7 = 0)

       resize a view with copying old data to new data at the corresponding indices

    .. cppkokkos:function:: realloc(const size_t n0 = 0, const size_t n1 = 0, const size_t n2 = 0, const size_t n3 = 0, const size_t n4 = 0, const size_t n5 = 0, const size_t n6 = 0, const size_t n7 = 0)

       resize a view with discarding old data


    .. rubric:: *Private* Members

    :member: typedef original_view_type internal_view_type;
    :member: internal_view_type internal_view;


.. rubric:: Free Functions

.. cppkokkos:function:: contribute(View<DT1, VP...>& dest, Kokkos::Experimental::ScatterView<DT2, LY, ES, OP, CT, DP> const& src)

   convenience function to perform final reduction of ScatterView
   results into a resultant View; may be called following |parallelReduce|_.


Example
-------

.. code-block:: cpp

    KOKKOS_INLINE_FUNCTION int foo(int i) { return i; }
    KOKKOS_INLINE_FUNCTION double bar(int i) { return i*i; }

    Kokkos::View<double*> results("results", 1);
    Kokkos::Experimental::ScatterView<double*> scatter(results);
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int input_i) {
        auto access = scatter.access();
        auto result_i = foo(input_i);
        auto contribution = bar(input_i);
        access(result_i) += contribution;
    });
    Kokkos::Experimental::contribute(results, scatter);
