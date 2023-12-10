``TeamHandleConcept``
=====================

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

TeamHandleConcept defines the concept for the ``member_type`` of ``TeamPolicy`` and ``TeamTask``.
The actual type is defined through the policies, but meets the following API.
Note that the specific classes are only part of the public API as provided by ``TeamPolicy`` and
``TeamTask`` and only their parts as defined by the ``TeamHandleConcept``.
The actual name of the class, as well as potential template parameters, existing
constructors, member functions going beyond the concept, etc., are NOT part of the public Kokkos API
and are thus subject to change.


Description
-----------

.. cppkokkos:class:: TeamHandleConcept


   .. rubric:: Public nested aliases

   .. cppkokkos:type:: execution_space

      Specifies the `execution space <https://kokkos.github.io/kokkos-core-wiki/API/core/execution_spaces.html>`_ associated to the team

   .. cppkokkos:type:: scratch_memory_space

      The scratch memory space associated to this team's execution space

   .. rubric:: Constructors

   .. cppkokkos:function:: TeamHandleConcept() = default

      Default constructor.

   .. cppkokkos:function:: TeamHandleConcept( TeamHandleConcept && ) = default

      Move constructor.

   .. cppkokkos:function:: TeamHandleConcept( TeamHandleConcept const & ) = default

      Copy constructor.

   .. cppkokkos:function:: ~TeamHandleConcept() = default

      Destructor.

   .. rubric:: Assignment

   .. cppkokkos:function:: TeamHandleConcept & operator = ( TeamHandleConcept && ) = default

      Move assignment.

   .. cppkokkos:function:: TeamHandleConcept & operator = ( TeamHandleConcept const & ) = default

      Assignment operators. Returns: ``*this``.

   .. rubric:: Index Queries

   .. cppkokkos:kokkosinlinefunction:: int team_rank() const noexcept ;

      Returns: the index ``i`` of the calling thread within the team with ``0 <= i < team_size()``

   .. cppkokkos:kokkosinlinefunction:: int team_size() const noexcept ;

      Returns: the number of threads associated with the team.

   .. cppkokkos:kokkosinlinefunction:: int league_rank() const noexcept ;

      Returns: the index ``i`` of the calling team within the league with ``0 <= i < league_size()``

   .. cppkokkos:kokkosinlinefunction:: int league_size() const noexcept ;

      Returns: the number of teams/workitems launched in the kernel.


   .. rubric:: Scratch Space Control

   .. cppkokkos:kokkosinlinefunction:: const scratch_memory_space & team_shmem() const ;

      Equivalent to calling ``team_scratch(0)``.

   .. cppkokkos:kokkosinlinefunction:: const scratch_memory_space & team_scratch(int level) const ;

      This function returns a scratch memory handle shared by all threads in a team, which allows access to scratch memory.
      This handle can be given as the first argument to a ``Kokkos::View`` to make it use scratch memory.

      - ``level``: The level of requested scratch memory is either ``0`` or ``1``.

      - Returns: a scratch memory handle to the team shared scratch memory specified by level.

   .. cppkokkos:kokkosinlinefunction:: const scratch_memory_space & thread_scratch(int level) const ;

      This function returns a scratch memory handle specific to the calling thread,
      which allows access to its private scratch memory. This handle can be given as the
      first argument to a ``Kokkos::View`` to make it use scratch memory.

      - ``level``: The level of requested scratch memory is either ``0`` or ``1``.

      - Returns: a scratch memory handle to the thread scratch memory specified by level.


   .. rubric:: Team Collective Operations

   The following functions must be called collectively by all members of a team.
   These calls must be lexically the same call, i.e. it is not legal to have some members of a team call
   a collective in one branch and the others in another branch of the code (see example).

   .. cppkokkos:kokkosinlinefunction:: void team_barrier() const noexcept ;

      All members of the team wait at the barrier, until the whole team arrived. This also issues a memory fence.

   .. cppkokkos:kokkosinlinefunction:: template<typename T> void team_broadcast( T & value , const int source_team_rank ) const noexcept;

      After this call ``var`` contains for every member of the team the value of ``var`` from the thread for which ``team_rank() == source_team_rank``.

      - ``var``: a variable of type ``T`` which gets overwritten by the value of ``var`` from the source rank.

      - ``source_team_rank``: identifies the broadcasting member of the team.

   .. cppkokkos:kokkosinlinefunction:: template<class Closure, typename T> void team_broadcast( Closure const & f , T & value , const int source_team_rank) const noexcept;

      After this call ``var`` contains for every member of the team the value of ``var`` from the thread for which ``team_rank() == source_team_rank`` after applying ``f``.

      - ``f``: a function object with an ``void operator() ( T & )`` which is applied to ``var`` before broadcasting it.

      - ``var``: a variable of type ``T`` which gets overwritten by the value of ``f(var)`` from the source rank.

      - ``source_team_rank``: identifies the broadcasting member of the team.

   .. cppkokkos:kokkosinlinefunction:: template< typename ReducerType> void team_reduce( ReducerType const & reducer ) const noexcept;

      Performs a reduction accross all members of the team as specified by ``reducer``. ``ReducerType`` must meet the concept of ``Kokkos::Reducer``.

   .. cppkokkos:kokkosinlinefunction:: template< typename T > T team_scan( T const & value , T * const global = 0 ) const noexcept;

      Performs an exclusive scan over the ``var`` provided by the team members. Let ``t = team_rank()`` and ``VALUES[t]`` the value of ``var`` from thread ``t``.

      - Returns: ``VALUES[0]`` + ``VALUES[1]`` + ``...`` + ``VALUES[t-1]`` or zero for ``t==0``.

      - ``global`` if provided will be set to ``VALUES[0]`` + ``VALUES[1]`` + ``...`` + ``VALUES[team_size()-1]``,
	must be the same pointer for every team member.

Examples
--------

.. code-block:: cpp

    typedef TeamPolciy<...> policy_type;
    parallel_for(policy_type(N,TEAM_SIZE).set_scratch_size(PerTeam(0,4096)),
                KOKKOS_LAMBDA (const typename policy_type::member_type& team_handle) {
        int ts = team_handle.team_size(); // returns TEAM_SIZE
        int tid = team_handle.team_rank(); // returns a number between 0 and TEAM_SIZE
        int ls = team_handle.league_size(); // returns N
        int lid = team_handle.league_rank(); // returns a number between 0 and N

        int value = tid * 5;
        team_handle.team_broadcast(value, 3);
        // value==15 on every thread
        value += tid;
        team_handle.team_broadcast([&] (int & var) { var*=2 }, value, 2);
        // value==34 on every thread
        int global;
        int scan = team_handle.team_scan(tid+1, &global);
        // scan == tid*(tid+1)/2 on every thread
        // global == ts*(ts-1)/2 on every thread
        Kokkos::View<int*, policy_type::execution_space::scratch_memory_type>
        a(team_handle.team_scratch(0), 1024);

    });
