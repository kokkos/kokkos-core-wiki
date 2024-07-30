``parallel_scan``
=================

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::parallel_scan( name, policy, functor, result );
    Kokkos::parallel_scan( name, policy, functor );
    Kokkos::parallel_scan( policy, functor, result);
    Kokkos::parallel_scan( policy, functor );

Dispatches parallel work defined by ``functor`` according to the *ExecutionPolicy* ``policy`` and perform a pre (exclusive) or post (inclusive) scan of the contributions
provided by the work items. The optional label ``name`` is used by profiling and debugging tools.  If provided, the final result is placed in result.

Interface
---------

.. cpp:function:: template <class ExecPolicy, class FunctorType> Kokkos::parallel_scan(const std::string& name, const ExecPolicy& policy, const FunctorType& functor);

.. cpp:function:: template <class ExecPolicy, class FunctorType> Kokkos::parallel_scan(const ExecPolicy&  policy, const FunctorType& functor);

.. cpp:function:: template <class ExecPolicy, class FunctorType, class ReturnType> Kokkos::parallel_scan(const std::string& name, const ExecPolicy&  policy, const FunctorType& functor, ReturnType&        return_value);

.. cpp:function:: template <class ExecPolicy, class FunctorType, class ReturnType> Kokkos::parallel_scan(const ExecPolicy&  policy, const FunctorType& functor, ReturnType&        return_value);

Parameters:
~~~~~~~~~~~

* ``name``: A user provided string which is used in profiling and debugging tools via the Kokkos Profiling Hooks.
* ExecPolicy: An *ExecutionPolicy* which defines iteration space and other execution properties. Valid policies are:

  - ``IntegerType``: defines a 1D iteration range, starting from 0 and going to a count.
  - `RangePolicy <../policies/RangePolicy.html>`_: defines a 1D iteration range.
  - `ThreadVectorRange <../policies/ThreadVectorRange.html>`_: defines a 1D iteration range to be executed through vector parallelization dividing the threads within a team.  Only valid inside a parallel region executed through a ``TeamPolicy`` or a ``TaskTeam``.
* FunctorType: A valid functor with (at minimum) an ``operator()`` with a matching signature for the ``ExecPolicy`` combined with the reduced type.
* ReturnType: a POD type with ``operator +=`` and ``operator =``, or a ``Kokkos::View``.

Requirements:
~~~~~~~~~~~~~

* The ``functor`` has a member function of the form ``operator() (const HandleType& handle, ReturnType& value, const bool final) const`` or ``operator() (const WorkTag, const HandleType& handle, ReturnType& value, const bool final) const``

  - The ``WorkTag`` free form of the operator is used if ``ExecPolicy`` is an ``IntegerType`` or ``ExecPolicy::work_tag`` is ``void``.
  - ``HandleType`` is an ``IntegerType`` if ``ExecPolicy`` is an ``IntegerType`` else it is ``ExecPolicy::member_type``.
* The type ``ReturnType`` of the ``functor`` operator must be compatible with the ``ReturnType`` of the parallel_scan and must match the arguments of the ``init`` and ``join`` functions of the functor if provided. If the functor doesn't have an ``init`` member function, it is assumed that the identity for the scan operation is given by the default constructor of the value type (and not by ``reduction_identity```).
* the functor must define FunctorType::value_type the same as ReturnType

Semantics
---------

* Neither concurrency nor order of execution are guaranteed.
* The ``ReturnType`` content will be overwritten, i.e. the value does not need to be initialized to the reduction-neutral element.
* The input value to the operator may contain a partial result, Kokkos may only combine the thread local contributions in the end. The operator should modify the input value according to the desired scan operation.
* It is not guaranteed that the functor will ever be called with ``final = false``.
* The functor might be called multiple times with ``final = false`` and the user has to make sure that the behavior in this case stays the same for repeated calls.

Examples
--------

.. code-block:: cpp

    #include<Kokkos_Core.hpp>
    #include<cstdio>

    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc,argv);
      {
        int N = argc>1?atoi(argv[1]):100;
        int64_t result;
        Kokkos::View<int64_t*>post("postfix_sum",N);
        Kokkos::View<int64_t*>pre("prefix_sum",N);

        Kokkos::parallel_scan("Loop1", N,
          KOKKOS_LAMBDA(int64_t i, int64_t& partial_sum, bool is_final) {
          if(is_final) pre(i) = partial_sum;
          partial_sum += i;
          if(is_final) post(i) = partial_sum;
        }, result);

        // pre (exclusive): 0,0,1,3,6,10,...
        // post (inclusive): 0,1,3,6,10,...
        // result: N*(N-1)/2
        printf("Result: %i %li\n",N,result);
      }
      Kokkos::finalize();
    }
