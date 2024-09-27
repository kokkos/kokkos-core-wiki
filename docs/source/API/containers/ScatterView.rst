``ScatterView``
===============

.. role:: cppkokkos(code)
	:language: cppkokkos

Header File: ``<Kokkos_ScatterView.hpp>``

Description
-----------
`Kokkos View-based <../core/view/view.html>`_ data structure that abstracts over "scatter - contribute" algorithms, where many contributors (over multiple indices) are reduced to fewer output resultants (*e.g.*, sum, product, maximum or minimum). ``ScatterView`` transparently switches between **Atomic**-,  and **Data Replication**-based scatter algorithms.  Typically, a ``ScatterView`` wraps an existing View. 


Interface
-----------

.. cppkokkos:class:: template <class DataType, class LayoutType, class ExecutionSpace, class Operation, class Duplication, class Contribution> ScatterView


Parameters
-----------

*  ``DataType``:  See `Kokkos View DataType <../core/view/view.html>`_

*  ``Layout``:  See  `Kokkos View LayoutType <../core/view/view.html>`_

*  ``ExecutionSpace``:  Where the code will be executed (CPU or GPU); typical values are ``Kokkos::DefaultHostExecutionSpace``, ``Kokkos::DefaultExecutionSpace``

*  ``Operation``:  ScatterSum, ScatterProd, ScatterMin, ScatterMax

*  ``Duplication``:  ScatterDuplicated, ScatterNonDuplicated

*  ``Contribution``:  ScatterAtomic, ScatterNonAtomic  

Public Class Members
--------------------

    .. cppkokkos:type:: original_view_type

        Type of View passed to ScatterView constructor.

    .. cppkokkos:type:: original_value_type

        Value type of the original_view_type.

    .. cppkokkos:type:: original_reference_type

        Reference type of the original_view_type.

    .. cppkokkos:type:: data_type_info

        DuplicatedDataType, a newly created DataType with a new runtime dimension that becomes the largest-stride dimension from the given View DataType.

    .. cppkokkos:type:: internal_data_type

        Value type of data_type_info.

    .. cppkokkos:type:: internal_view_type

        Type alias for a View type created from the internal_data_type.

Public Class Methods
--------------------

    .. cppkokkos:function:: access() const

       use within a kernel to return a ``ScatterAccess`` member; this member accumulates a given thread's contribution to the reduction.

    .. cppkokkos:function:: subview() const

        :return: a subview of a ``ScatterView``

    .. cppkokkos:function:: contribute_into(View<DT, RP...> const& dest) const

       contribute ``ScatterView`` array's results into the input View ``dest``

    .. cppkokkos:function:: reset()

       performs reset on destination array

    .. cppkokkos:function:: reset_except(View<DT, RP...> const& view)

       excludes a Kokkos View from reset

    .. cppkokkos:function:: resize(const size_t n0 = 0, const size_t n1 = 0, const size_t n2 = 0, const size_t n3 = 0, const size_t n4 = 0, const size_t n5 = 0, const size_t n6 = 0, const size_t n7 = 0)

       resize a view with copying old data to new data at the corresponding indices

    .. cppkokkos:function:: realloc(const size_t n0 = 0, const size_t n1 = 0, const size_t n2 = 0, const size_t n3 = 0, const size_t n4 = 0, const size_t n5 = 0, const size_t n6 = 0, const size_t n7 = 0)

       resize a view with discarding old data


Constructors
-------------

    .. cppkokkos:function:: ScatterView()

        The default constructor. Default constructs members.

    .. cppkokkos:function:: ScatterView(View<RT, RP...> const&)

        Constructor from a ``Kokkos::View``. ``internal_view`` member is copy constructed from this input view.

    .. cppkokkos:function:: ScatterView(std::string const& name, Dims ... dims)

        Constructor from variadic pack of dimension arguments. Constructs ``internal_view`` member.

    .. cppkokkos:function:: ScatterView(ALLOC_PROP const& arg_prop, Dims... dims)

        Constructor from variadic pack of dimension arguments. Constructs ``internal_view`` member.
        This constructor allows passing an object created by ``Kokkos::view_alloc`` as first argument, e.g., for specifying an execution space via
        ``Kokkos::view_alloc(exec_space, "label")``.


Free Functions
--------------------


.. cppkokkos:function:: contribute(View<DT1, VP...>& dest, Kokkos::Experimental::ScatterView<DT2, LY, ES, OP, CT, DP> const& src)

   convenience function to perform final reduction of ScatterView
   results into a resultant View; may be called following `parallel_reduce <../core/parallel-dispatch/parallel_reduce.html>`_ .


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
