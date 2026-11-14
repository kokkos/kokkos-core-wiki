``SinglePolicy``
================

.. _single: ../parallel-dispatch/single.html

.. |single| replace:: ``Kokkos::single``


Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::SinglePolicy<...>()
    Kokkos::SinglePolicy<...>(exec_space)

    // CTAD Constructor
    Kokkos::SinglePolicy()

``SinglePolicy`` defines an execution policy that executes a functor or lambda exactly once.
It can only be used with the pattern |single|_ to perform the execution once, with the possibility of returning an output.

Synopsis
--------

.. code-block:: cpp

    template <typename... Properties>
    struct Kokkos::SinglePolicy {
        using traits = Kokkos::Impl::PolicyTraits<Properties...>;
        using execution_policy = Kokkos::SinglePolicy<Properties...>;
        using execution_space  = typename traits::execution_space;

        // Inherited from PolicyTraits<Args...>
        using execution_space = PolicyTraits<Args...>::execution_space;

        // Constructors
        SinglePolicy(const SinglePolicy&) = default;

        SinglePolicy();

        SinglePolicy(const execution_space& exec_space);

        // return ExecSpace instance provided to the constructor
        inline const execution_space & space() const;
    };

Parameters
----------

General Template Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Valid template arguments for ``SinglePolicy`` are described `here <../Execution-Policies.html#common-arguments-for-all-execution-policies>`_

The only exception is that the ``LaunchBounds`` argument is incompatible with ``SinglePolicy``.


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


Execute a functor once with zero arguments. The functor can also be called with a WorkTag to choose which `operator()` will be called.

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

    // With an ExecutionSpace instance
    Kokkos::DefaultExecutionSpace exec_space;
    Kokkos::single("label", Kokkos::SinglePolicy(exec_space), f);

    // With both a WorkTag and an ExecutionSpace
    Kokkos::single("label",
        Kokkos::SinglePolicy<TimesTwoTag, Kokkos::DefaultExecutionSpace>(), f);


