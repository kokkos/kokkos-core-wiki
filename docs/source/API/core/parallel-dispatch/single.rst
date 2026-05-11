``single``
==========

.. _SinglePolicy: ../policies/SinglePolicy.html

.. |SinglePolicy| replace:: ``SinglePolicy``

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::single(name, policy, functor);
    Kokkos::single(name, policy, functor, output...);
    Kokkos::single(policy, functor);
    Kokkos::single(policy, functor, output...);
    Kokkos::single(functor);
    Kokkos::single(name, functor);

Execute functor on restricted resource defined by the policy

Interface
---------

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType> 
    Kokkos::single(const std::string& name, 
                   const ExecPolicy& policy, 
                   const FunctorType& functor);

.. code-block:: cpp

    template <class ExecPolicy, class FunctorType, class ValueType> 
    Kokkos::single(const ExecPolicy& policy, 
                   const FunctorType& functor, 
                   ValueType& val);

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

* ``ExecPolicy``: defines execution properties, valid policies are:

  - :cpp:func:`SinglePolicy` restricts execution to a single thread in the execution space.
  - :cpp:func:`PerTeam` restricts execution to a single vector lane in the calling team.
  - :cpp:func:`PerThread` restricts execution to a single vector lane in the calling thread.

* ``name`` is only usable with |SinglePolicy|_

* ``val`` is a reference to the output variable. The functor's ``operator()`` receives a reference to the reduction value.

Examples
--------

Using ``Kokkos::single`` in Hierarchical Parallelism

.. code-block:: cpp

  #include<Kokkos_Core.hpp>

  int main(int argc, char* argv[]) {
    Kokkos::initialize(argc,argv);

    using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    using team_handle = team_policy::member_type;

    Kokkos::parallel_for(team_policy(100, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_handle& team) {
        int val;
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          val = team.league_rank();
        });
    });

    Kokkos::finalize();
    return 0;
  }


Using ``Kokkos::single`` to execute a functor once and produce a single output

.. code-block:: cpp

  #include<Kokkos_Core.hpp>

  int main(int argc, char* argv[]) {
    Kokkos::initialize(argc,argv);

    float value;
    Kokkos::single("label", Kokkos::SinglePolicy(),
      KOKKOS_LAMBDA(float& val) {
        val = 1.0;
    }, value);

    Kokkos::finalize();
    return 0;
  }


