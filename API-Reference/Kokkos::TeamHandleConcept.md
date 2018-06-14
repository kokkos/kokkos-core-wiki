# `Kokkos::TeamPolicy`

Header File: `Kokkos_Core.hpp`

TeamHandleConcept defines the concept for the `member_type` of `TeamPolicy` and `TeamTask`.
The actual type is defined through the policies, but meets the following API requirements

# Synopsis 
  ```c++
  class TeamHandleConcept {
    // Constructor, Destructor, Assignment
    TeamHandleConcept( TeamHandleConcept && ) = default ;
    TeamHandleConcept( TeamHandleConcept const & ) = default ;
    ~TeamHandleConcept() = default ;
    TeamHandleConcept & operator = ( TeamHandleConcept && ) = default ;
    TeamHandleConcept & operator = ( TeamHandleConcept const & ) = default ;

    // Indicies
    KOKKOS_INLINE_FUNCTION
    int team_rank() const noexcept;
    KOKKOS_INLINE_FUNCTION
    int team_size() const noexcept;
    KOKKOS_INLINE_FUNCTION
    int league_rank() const noexcept;
    KOKKOS_INLINE_FUNCTION
    int league_size() const noexcept;

    // Scratch Space
    KOKKOS_INLINE_FUNCTION
    const scratch_memory_space & team_shmem() const;
    KOKKOS_INLINE_FUNCTION
    const scratch_memory_space & team_scratch(int level) const;
    KOKKOS_INLINE_FUNCTION
    const scratch_memory_space & thread_scratch(int level) const;

    // Team collectives
    KOKKOS_INLINE_FUNCTION 
    void team_barrier() const noexcept;
    template<typename T>
    KOKKOS_INLINE_FUNCTION
    void team_broadcast( T & value , const int source_team_rank ) const noexcept;
    template<class Closure, typename T>
    KOKKOS_INLINE_FUNCTION
    void team_broadcast( Closure const & f , T & value , const int source_team_rank) const noexcept;
    template< typename ReducerType >
    KOKKOS_INLINE_FUNCTION
    void team_reduce( ReducerType const & reducer ) const noexcept;
    template< typename T >
    KOKKOS_INLINE_FUNCTION
    T team_scan( T const & value , T * const global = 0 ) const noexcept;
  };
  ```

## Public Class Members

### Constructors
 
 * TeamPolicy(): Default Constructor unitialized policy.
 * ```c++
   TeamPolicy(index_type league_size, index_type team_size [, index_type vector_length])
   ```
   Request to launch `league_size` work items, each of which is assigned to a team of threads with `team_size` threads, using a vector length of `vector_length`.
   If the team size is not possible when calling a parallel policy, that kernel launch may throw. 
 
 * ```c++
   TeamPolicy(index_type league_size, Impl::AUTO_t [, index_type vector_length])
   ```
   Request to launch `league_size` work items, each of which is assigned to a team of threads of a size determined by Kokkos, using a vector length of `vector_length`.
   The team size may be determined lazely at launch time, taking into account properties of the functor.

 * ```c++
   TeamPolicy(execution space, index_type league_size, index_type team_size [, index_type vector_length])
   ```
   Request to launch `league_size` work items, each of which is assigned to a team of threads with `team_size` threads, using a vector length of `vector_length`.
   If the team size is not possible when calling a parallel policy, that kernel launch may throw. 
   Use the provided execution space instance during a kernel launch.  

 * ```c++
   TeamPolicy(execution_space space, index_type league_size, Impl::AUTO_t [, index_type vector_length])
   ```
   Request to launch `league_size` work items, each of which is assigned to a team of threads of a size determined by Kokkos, using a vector length of `vector_length`.
   The team size may be determined lazely at launch time, taking into account properties of the functor.
   Use the provided execution space instance during a kernel launch.  

### Runtime Settings

  * ```c++
    inline TeamPolicy& set_chunk_size(int chunk);
    ```
    Set the chunk size. Each physical team of threads will get assigned `chunk` consecutive teams. Default is 1.

  * ```c++
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team, const Impl::PerThreadValue& per_thread);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread, const Impl::PerTeamValue& per_team);
    ```
    Set the per team and per thread scratch size. 
    * `level`: set the storage level. 0 is closest cache. 1 is closest storage (e.g. high bandwidth memory)
    * `per_team`: wrapper for the per team size of scratch in bytes. Returned by the function `PerTeam(int)`.
    * `per_thread`: wrapper for the per thread size of scratch in bytes. Returned by the function `PerThread(int)`.
    One can set the scratch size for level 0 and 1 independently by calling the function twice. 
    Subsequent calls with the same level overwrite the previous value. 

## Examples

  ```c++
    TeamPolicy<> policy_1(N,AUTO);
    TeamPolicy<Cuda> policy_2(N,T);
    TeamPolicy<Schedule<Dynamic>, OpenMP> policy_3(N,AUTO,8);
    TeamPolicy<IndexType<int>, Schedule<Dynamic>> policy_4(N,1,4);
    TeamPolicy<OpenMP> policy_5(OpenMP(), N, AUTO);
  ```

