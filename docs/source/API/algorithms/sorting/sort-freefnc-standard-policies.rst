
.. role:: cppkokkos(code)
    :language: cppkokkos

Sort
====

Header File: ``<Kokkos_Sort.hpp>``.

This page describes the Kokkos' sorting API to use with standard host-based dispatching.

API
^^^

Overload set using ``operator<`` for comparing elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cppkokkos:function:: template <class ExecutionSpace, class DataType, class... Properties> void sort(const ExecutionSpace& exec, const Kokkos::View<DataType, Properties...>& view);

   Sort the elements in ``view`` in non-discending order using the provided ``exespace``.
   Elements are compared using ``operator<``.

   :param exec: execution space instance
   :param view: view to sort

   Semantics:

   - this function is potentially asynchronous. If needed, you can block either using
     a `global fence <../core/parallel-dispatch/fence.html>`_ or you can just use `fence
     the execution space instance <../core/execution_spaces.html>`_ argument
     you pass to the function

   Constraints:

   - ``view`` must be rank-1 with ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

   - ``view`` must be accessible from ``exespace``


.. cppkokkos:function:: template <class DataType, class... Properties> void sort(const Kokkos::View<DataType, Properties...>& view);

   Sort the elements in ``view`` in non-discending order using the view's associated execution space.
   Elements are compared using ``operator<``.

   :param view: view to sort

   Constraints

   - ``view`` must be rank-1 with ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

   Semantics:

   - This is a blocking function. Before the sorting is executed, the function internally
     calls ``Kokkos::fence("Kokkos::sort: before")``, and after the sort is completed
     the execution space is fenced with message ``exec.fence("Kokkos::sort: fence after sorting")``.

   Possible implementation:

   .. code-block:: cpp

      template <class DataType, class... Properties>
      void sort(const Kokkos::View<DataType, Properties...>& view){
        using ViewType = Kokkos::View<DataType, Properties...>;
        typename ViewType::execution_space exec;
	sort(exec, view);
      }

Overload set accepting a custom comparison object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2.0

.. cppkokkos:function:: template <class ExecutionSpace, class ComparatorType, class DataType, class... Properties> void sort(const ExecutionSpace& exec, const Kokkos::View<DataType, Properties...>& view, const ComparatorType& comparator)

   Sort the elements in ``view`` in non-discending order using the view's associated execution space.
   Elements are compared using the comparison functor ``comparator``.

   :param exec: execution space instance
   :param view: view to sort
   :param comparator: comparison functor

   Constraints:

   - ``view`` must be rank-1 with ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

   - ``view`` must be accessible from ``exespace``

   - ``comparator``: comparison function object returning ``true`` if the first
     argument is less than the second. Must be valid to be called from the execution space passed,
     and callable with two arguments ``a,b`` of type (possible const-qualified) ``value_type``,
     where ``value_type`` is the non-const value type of the view. Must conform to:

     .. code-block:: cpp

	template <class T>
	struct CustomComparison {
	  KOKKOS_FUNCTION bool operator()(T a, T b) const{
	    // return true if a is less than b
	  }
	};

   Semantics:

   - this function is potentially asynchronous. If needed, you can block either using
     a `global fence <../core/parallel-dispatch/fence.html>`_ or you can just use `fence
     the execution space instance <../core/execution_spaces.html>`_ argument
     you pass to the function


.. cppkokkos:function:: template <class ComparatorType, class DataType, class... Properties> void sort(const Kokkos::View<DataType, Properties...>& view, const ComparatorType& comparator)

   Sort the elements in ``view`` in non-discending order using the view's associated execution space.
   Elements are compared using the comparison functor ``comparator``.

   :param view: view to sort
   :param comparator: comparison functor

   Constraints:

   - ``view`` must be rank-1 with ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

   - ``comparator``: same requirements as overload above

   Semantics:

   - This is a blocking function. Before the actual sorting is executed, the function
     calls ``Kokkos::fence("Kokkos::sort with comparator: before)"``; after the sort is completed
     the execution space is fenced with
     message ``exec.fence("Kokkos::sort with comparator: fence after sorting")``.

   Possible implementation:

   .. code-block:: cpp

      template <class ComparatorType, class DataType, class... Properties>
      void sort(const Kokkos::View<DataType, Properties...>& view,
                const ComparatorType& comparator)
      {
        using ViewType = Kokkos::View<DataType, Properties...>;
        typename ViewType::execution_space exec;
	sort(exec, view, comparator);
      }

Overload set to sort a View's subrange
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cppkokkos:function:: template <class ExecutionSpace, class ViewType> void sort(const ExecutionSpace& exec, ViewType view, size_t const startIndex, size_t const endIndex)

   Sort a subrange of elements of ``view`` in non-discending order using the given execution space.

   :param exec: execution space instance
   :param view: view to sort
   :param startIndex, endIndex: indices representing the range of elements to sort (``endIndex`` is exclusive)

   Constraints:

   - ``view`` must be rank-1

   Preconditions:

   - ``startIndex, endIndex`` must represent a valid range, i.e., ``endIndex >= startIndex``, and be admissible for the given ``view``, i.e., ``endIndex < view.extent(0)``

   Semantics:

   - this function is potentially asynchronous. If needed, you can block either using
     a `global fence <../core/parallel-dispatch/fence.html>`_ or you can just use `fence
     the execution space instance <../core/execution_spaces.html>`_ argument
     you pass to the function


.. cppkokkos:function:: template<class ViewType> void sort(ViewType view, size_t const startIndex, size_t const endIndex)

   Sort a subrange of elements of ``view`` in non-discending order using the view's associated execution space.

   :param view: view to sort
   :param startIndex, endIndex: indices representing the range of elements to sort (``endIndex`` is exclusive)

   Constraints: same as overload above

   Semantics:

   - This is a blocking function. ``Kokkos::fence("Kokkos::sort: before")`` is called before sorting,
     and the execution space is fenced after the sort with message ``Kokkos::Sort: fence after sorting``.

   Possible implementation:

   .. code-block:: cpp

      template <class ViewType>
      void sort(ViewType view, size_t const startIndex, size_t const endIndex) {
	typename ViewType::execution_space exec;
	sort(exec, view, startIndex, endIndex);
      }
