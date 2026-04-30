``SinglePolicy``
================

.. _single: ../parallel-dispatch/single.html

.. |single| replace:: ``Kokkos::single``

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::SinglePolicy<...>()
    Kokkos::SinglePolicy<...>(exec_space)

    // CTAD Constructor
    Kokkos::SinglePolicy()

``SinglePolicy`` defines an execution policy that executes a functor or lambda exactly once.
It has to be used with |single|_ to perform a single operation or to produce one
or more scalar outputs.

Synopsis
--------

.. code-block:: cpp

    template<class ... Args>
    struct Kokkos::SinglePolicy {
        using execution_policy = SinglePolicy;

        // Inherited from PolicyTraits<Args...>
        using execution_space   = PolicyTraits<Args...>::execution_space;
        using work_tag          = PolicyTraits<Args...>::work_tag;

        // Constructors
        SinglePolicy(const SinglePolicy&) = default;
        SinglePolicy(SinglePolicy&&) = default;

        SinglePolicy();

        SinglePolicy(const execution_space& exec_space);

        // return ExecSpace instance provided to the constructor
        KOKKOS_FUNCTION const execution_space & space() const;
    };

Parameters
----------

General Template Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Valid template arguments for ``SinglePolicy`` are described `here <../Execution-Policies.html#common-arguments-for-all-execution-policies>`_

Public Class Members
--------------------

Constructors
~~~~~~~~~~~~

.. cpp:function:: SinglePolicy()

   Default Constructor. Uses the default execution space.

.. cpp:function:: SinglePolicy(const ExecutionSpace& exec_space)

    Constructor that takes an ``ExecutionSpace`` instance.

Examples
--------

Using ``SinglePolicy`` for a single operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Execute a functor once with zero arguments. The functor can also be called with a WorkTag.

.. code-block:: cpp

    struct Functor {
      Kokkos::View<double> v;
      KOKKOS_FUNCTION void operator()() const { v() *= 3; }
      KOKKOS_FUNCTION void operator()(const TimesTwoTag) const { v() *= 2; }
    };

    Kokkos::View<double> v("v");
    Functor f{v};

    // Default execution space
    Kokkos::single("label", Kokkos::SinglePolicy(), f);

    // With an ExecutionSpace
    Kokkos::single("label",
        Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(), f);

    // With both a WorkTag and an ExecutionSpace
    Kokkos::single("label",
        Kokkos::SinglePolicy<TimesTwoTag, Kokkos::DefaultExecutionSpace>(), f);


Using ``SinglePolicy`` to produce scalar output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Produce a single scalar output by passing a reference to the output variable as an argument to the functor's ``operator()``. 
The functor can also be called with a WorkTag. The functor's ``operator()`` receives a reference to the reduction value.

.. code-block:: cpp

    struct ReductionFunctor {
      KOKKOS_FUNCTION void operator()(int& res) const { res = 5; }
      KOKKOS_FUNCTION void operator()(const TenTag, int& res) const { res = 10; }
    };

    int val;
    ReductionFunctor f;

    // With a WorkTag and an ExecutionSpace
    Kokkos::single("label",
        Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace, TenTag>(), f, val);
    // val == 10

    // With a lambda
    Kokkos::single("label",
        Kokkos::SinglePolicy<Kokkos::DefaultExecutionSpace>(),
        KOKKOS_LAMBDA(int& ret) { ret = 5; }, val);
    // val == 5

    // Minimal (default execution space, no work tag)
    Kokkos::single("label", Kokkos::SinglePolicy(), f, val);
    // val == 5


Using ``SinglePolicy`` to produce multiple outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The functor's ``operator()`` can receive multiple reduction arguments.

.. code-block:: cpp

    int val1, val2;

    // With a lambda producing 2 outputs
    Kokkos::single("label", Kokkos::SinglePolicy(),
        KOKKOS_LAMBDA(int& s1, int& s2) {
          s1 = 1;
          s2 = 2;
        }, val1, val2);

