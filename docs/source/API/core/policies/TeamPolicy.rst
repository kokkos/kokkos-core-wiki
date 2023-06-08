``TeamPolicy``
==============

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::TeamPolicy<>( league_size, team_size [, vector_length])
    Kokkos::TeamPolicy<ARGS>(league_size, team_size [, vector_length])
    Kokkos::TeamPolicy<>(Space, league_size, team_size [, vector_length])
    Kokkos::TeamPolicy<ARGS>(Space, league_size, team_size [, vector_length])

Execution policy for a 1D iteration space starting at begin and going to end with an open interval.

See also: `TeamMember <TeamHandleConcept.html>`_

Description
-----------

.. cppkokkos:class:: template<class ...Args> TeamPolicy

   .. rubric:: Template Arguments

   Valid template arguments for TeamPolicy are described `here <../Execution-Policies.html#common-arguments-for-all-execution-policies>`_

   .. rubric:: Public nested typedefs

   .. cppkokkos:type:: execution_space
   .. cppkokkos:type:: schedule_type
   .. cppkokkos:type:: work_tag
   .. cppkokkos:type:: index_type
   .. cppkokkos:type:: iteration_pattern
   .. cppkokkos:type:: launch_bounds
   .. cppkokkos:type:: member_type


   .. rubric:: Constructors

   .. cppkokkos:function:: TeamPolicy()

      Default constructor uninitialized policy.

   .. cppkokkos:function:: TeamPolicy(const TeamPolicy&) = default;

      Copy constructor

   .. cppkokkos:function:: TeamPolicy(TeamPolicy&&) = default;

      Move constructor

   .. cppkokkos:function:: TeamPolicy(index_type league_size, index_type team_size, index_type vector_length=1)

      Request to launch ``league_size`` work items, each of which is assigned to a team of threads
      with ``team_size`` threads, using a vector length of ``vector_length``. If the team size is not possible when
      calling a parallel policy, that kernel launch may throw.

   .. cppkokkos:function:: TeamPolicy(index_type league_size, Impl::AUTO_t, index_type vector_length=1)

      Request to launch ``league_size`` work items, each of which is assigned to a team of threads of a
      size determined by Kokkos, using a vector length of ``vector_length``. The team size may be determined
      lazily at launch time, taking into account properties of the functor.

   .. cppkokkos:function:: TeamPolicy(execution_space space, index_type league_size, index_type team_size, index_type vector_length=1)

      Request to launch ``league_size`` work items, each of which is assigned to a team of threads with ``team_size`` threads,
      using a vector length of ``vector_length``. If the team size is not possible when calling a parallel policy,
      that kernel launch may throw. Use the provided execution space instance during a kernel launch.

   .. cppkokkos:function:: TeamPolicy(execution_space space, index_type league_size, Impl::AUTO_t, index_type vector_length=1)

      Request to launch ``league_size`` work items, each of which is assigned to a team of threads of a size determined by Kokkos,
      using a vector length of ``vector_length``. The team size may be determined lazily at launch time, taking into
      account properties of the functor. Use the provided execution space instance during a kernel launch.

   .. rubric:: Runtime Settings

   .. cppkokkos:function:: inline TeamPolicy& set_chunk_size(int chunk);

      Set the chunk size. Each physical team of threads will get assigned ``chunk`` consecutive teams. Default is 1.

      Returns: reference to ``*this``

   .. cppkokkos:function:: inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team);

   .. cppkokkos:function:: inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread);

   .. cppkokkos:function:: inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team, const Impl::PerThreadValue& per_thread);

   .. cppkokkos:function:: inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread, const Impl::PerTeamValue& per_team);

      Set the per team and per thread scratch size.

      - ``level``: set the storage level. 0 is closest cache. 1 is closest storage (e.g. high bandwidth memory)

      - ``per_team``: wrapper for the per team size of scratch in bytes. Returned by the function ``PerTeam(int)``.

      - ``per_thread``: wrapper for the per thread size of scratch in bytes. Returned by the function ``PerThread(int)``.

      One can set the scratch size for level 0 and 1 independently by calling the function twice. Subsequent calls with the same level overwrite the previous value.
      Returns: reference to ``*this``

   .. rubric:: Query Limits of Runtime Settings

   .. _parallelFor: ../parallel-dispatch/parallel_for.html

   .. |parallelFor| replace:: :cppkokkos:func:`parallel_for`

   .. _parallelReduce: ../parallel-dispatch/parallel_reduce.html

   .. |parallelReduce| replace:: :cppkokkos:func:`parallel_reduce`

   .. cppkokkos:function:: template<class FunctorType> int team_size_max(const FunctorType& f, const ParallelForTag&) const;

   .. cppkokkos:function:: template<class FunctorType> int team_size_max(const FunctorType& f, const ParallelReduceTag&) const;

      Query the maximum team size possible given a specific functor. The tag denotes whether this is for a |parallelFor|_ or a |parallelReduce|_.
      Note: this is not a static function! The function will take into account settings for vector length and scratch size of ``*this``. Using a value larger than the return value will result in dispatch failure.
      Returns: The maximum value for ``team_size`` allowed to be given to be used with an otherwise identical ``TeamPolicy`` for dispatching the functor ``f``.

   .. cppkokkos:function:: template<class FunctorType> int team_size_recommended(const FunctorType& f, const ParallelForTag&) const;

   .. cppkokkos:function:: template<class FunctorType> int team_size_recommended(const FunctorType& f, const ParallelReduceTag&) const;

      Query the recommended team size for the specific functor ``f``. The tag denotes whether this is for a |parallelFor|_ or a |parallelReduce|_.
      Note: this is not a static function! The function will take into account settings for vector length and scratch size of ``*this``.
      Returns: The recommended value for ``team_size`` to be given to be used with an otherwise identical ``TeamPolicy`` for dispatching the functor ``f``.

   .. cppkokkos:function:: static int vector_length_max();

      Returns: the maximum valid value for vector length.

   .. cppkokkos:function:: static int scratch_size_max(int level);

      Returns: the maximum total scratch size in bytes, for the given level.
      Note: If a kernel performs team-level reductions or scan operations, not all of this memory will be
      available for dynamic user requests. Some of that maximal scratch size is being used for internal operations.
      The actual size of these internal allocations depends on the value type used in the reduction or scan.

   .. rubric:: Query Runtime Settings

   .. cppkokkos:function:: int team_size() const;

      Returns: the requested team size.

   .. cppkokkos:function:: int league_size() const;

      Returns: the requested league size.

   .. cppkokkos:function:: int scratch_size(int level, int team_size_ = -1) const;

      This function returns the total scratch size requested. If ``team_size`` is not provided, the team size
      for the calculation is used from the internal setting (i.e. the result of calling ``this->team_size()``). Otherwise, the provided team size is used.
      Returns: the value for the total scratch size in bytes in the specified scratch level.

   .. cppkokkos:function:: int team_scratch_size(int level) const;

      Returns: the value for the per team scratch size in bytes in the specified scratch level.

   .. cppkokkos:function:: int thread_scratch_size(int level) const;

      Returns: the value for the per thread scratch size in bytes in the specified scratch level.

   .. cppkokkos:function:: int chunk_size() const;

      Returns: the chunk size, set via ``set_chunk_size()``.

Examples
--------

.. code-block:: cpp

    TeamPolicy<> policy_1(N,AUTO);
    TeamPolicy<Cuda> policy_2(N,T);
    TeamPolicy<Schedule<Dynamic>, OpenMP> policy_3(N,AUTO,8);
    TeamPolicy<IndexType<int>, Schedule<Dynamic>> policy_4(N,1,4);
    TeamPolicy<OpenMP> policy_5(OpenMP(), N, AUTO);
