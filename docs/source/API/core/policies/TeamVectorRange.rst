``TeamVectorRange``
===================

.. role::cpp(code)
   :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   parallel_for(TeamVectorRange(team,range), [=] (int i) {...});
   parallel_reduce(TeamVectorRange(team,begin,end),
       [=] (int i, double& lsum) {...},sum);

TeamVectorRange is a `nested execution policy <NestedPolicies.html>`_ used inside hierarchical parallelism.
In contrast to global policies, the public interface for nested policies is implemented
as functions, in order to enable implicit templating on the execution space type via
the team handle.

Synopsis
--------

.. code-block:: cpp

   template<class TeamMemberType, class iType>
   /* implementation defined */ TeamVectorRange(TeamMemberType team, iType count);
   template<class TeamMemberType, class iType1, class iType2>
   /* implementation defined */ TeamVectorRange(TeamMemberType team, iType1 begin, iType2 end);

Description
-----------

.. code-block:: cpp

   template<class TeamMemberType, class iType>
   /* implementation defined */ TeamVectorRange(TeamMemberType team, iType count);

Splits the index range ``0`` to ``count-1`` over the threads of the team and their vector lanes.

* **Arguments**
    - ``team``: a handle to the calling team execution context.
    - ``count``: index range length.

* **Returns**
    - Implementation defined type.

* **Requirements**
    - ``TeamMemberType`` is a type that models `TeamHandle <TeamHandleConcept.html>`_
    - ``std::is_integral<iType>::value`` is true.
    - Every member thread of ``team`` must call the operation in the same branch, i.e. it is not legal to have some
        threads call this function in one branch, and the other threads of ``team`` call it in another branch.
    - ``count >= 0`` is true;

.. code-block:: cpp

   template<class TeamMemberType, class iType1, class iType2>
   /* implementation defined */ TeamVectorRange(TeamMemberType team, iType1 begin, iType2 end);

Splits the index range ``begin`` to ``end-1`` over the threads of the team and their vector lanes.

* **Arguments**
    - ``team``: a handle to the calling team execution context.
    - ``begin``: index range begin.
    - ``end``: index range end.

* **Returns**
    - Implementation defined type.

* **Requirements**
    - ``TeamMemberType`` is a type that models `TeamHandle <TeamHandleConcept.html>`_
    - ``std::is_integral<iType1>::value`` is true.
    - ``std::is_integral<iType2>::value`` is true.
    - Every member thread of ``team`` must call the operation in the same branch, i.e. it is not legal to have some threads call this function in one branch, and the other threads of ``team`` call it in another branch..
    - ``end >= begin`` is true;

Examples
--------

.. code-block:: cpp

   typedef TeamPolicy<>::member_type team_handle;
   parallel_for(TeamPolicy<>(N,AUTO,4), KOKKOS_LAMBDA (const team_handle& team) {
       int n = team.league_rank();
       parallel_for(TeamVectorRange(team,M), [&] (const int& i) {
           A(n,i) = B(n) + i;
       });
       team.team_barrier();
       int team_sum;
       parallel_reduce(TeamVectorRange(team,M), [&] (const int& i, int& lsum) {
           lsum += A(n,i);
       },team_sum);
       single(PerTeam(team),[&] () {
           A_rowsum(n) += team_sum;
       });
   });
