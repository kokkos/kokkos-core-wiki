``single``
==========

.. _SinglePolicy: ../policies/SinglePolicy.html

.. |SinglePolicy| replace:: ``SinglePolicy``


Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::single(name, policy, functor);
    Kokkos::single(name, policy, functor, output);
    Kokkos::single(policy, functor);
    Kokkos::single(policy, functor, output);
    Kokkos::single(functor);
    Kokkos::single(name, functor);

Execute the functor exactly once on the execution space defined by the policy

Interface
---------

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType> 
    Kokkos::single(const std::string& name, 
                   const ExecPolicy& policy, 
                   const FunctorType& functor);

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType, class ReturnType> 
    Kokkos::single(const ExecPolicy& policy, 
                   const FunctorType& functor, 
                   ReturnType& ret);

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType> 
    Kokkos::single(const ExecPolicy& policy, const FunctorType& functor);

.. code-block:: cpp

    template <class FunctorType> 
    Kokkos::single(const std::string& name, const FunctorType& functor);

.. code-block:: cpp

    template <class FunctorType> 
    Kokkos::single(const FunctorType& functor);


Parameters:
~~~~~~~~~~~

* ``name``: A user provided string which is used in profiling and debugging tools via the Kokkos Profiling Hooks. 

* ``ExecPolicy``: Defines execution properties, valid policies are:

  - :cpp:struct:`SinglePolicy` restricts execution to a single thread in the execution space.
  - :cpp:func:`PerTeam` restricts execution to a single vector lane in the calling team.
  - :cpp:func:`PerThread` restricts execution to a single vector lane in the calling thread.

*  ``FunctorType``: A valid functor having either a ``operator() () const`` or ``operator() (ReturnType&) const`` member function. The functor can be defined using a C++ class/struct or a lambda.

* ``ret``: A reference to the output variable. The functor's ``operator()`` receives a reference to this value.

Requirements
~~~~~~~~~~~~
* ``policy`` should be defined with no ``LaunchBounds`` attribute.

* if ``single`` is called without a return value, ``FunctorType`` needs a member function with signature ``operator() () const``.

* if ``single`` is called with a return value, ``FunctorType`` needs a member function with signature ``operator() (ReturnType&) const`` instead.

Semantics
---------

* The functors call operator is invoked exactly once without any parallelization.

* The call is potentially asynchronous. To guarantee a kernel has finished, a developer should call fence on the execution space on which the kernel is run.

Examples
--------

Using ``Kokkos::single`` in Hierarchical Parallelism

.. code-block:: cpp

  #include<Kokkos_Core.hpp>

  int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc,argv);

    using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using team_handle = team_policy::member_type;

    Kokkos::parallel_for(team_policy(100, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_handle& team) {
        int val;
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          val = team.league_rank();
        });
    });

    return 0;
  }


Using ``Kokkos::single`` to execute a functor once and produce a scalar output

.. warning::
   The OpenACC backend currently doesn't support returning a value with ``single``

.. code-block:: cpp

  #include<Kokkos_Core.hpp>

  int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc,argv);

    float value;
    Kokkos::single("label", Kokkos::SinglePolicy(),
      KOKKOS_LAMBDA(float& val) {
        val = 1.0;
    }, value);

    return 0;
  }


To produce several outputs using ``Kokkos::single``, a custom return type needs to be used

.. warning::
   The OpenACC backend currently doesn't support returning a value with ``single``

.. code-block:: cpp

  #include<Kokkos_Core.hpp>

  struct CustomReturn {
    int val1;
    int val2;
  };

  int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard guard(argc,argv);

    CustomReturn ret;

    Kokkos::single("label", Kokkos::SinglePolicy(),
        KOKKOS_LAMBDA(CustomReturn& r) {
          r.val1 = 1;
          r.val2 = 2;
        }, ret);

    return 0;
  }

