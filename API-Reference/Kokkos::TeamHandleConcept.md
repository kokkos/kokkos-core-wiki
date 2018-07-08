# `Kokkos::TeamHandleConcept`

Header File: `Kokkos_Core.hpp`

TeamHandleConcept defines the concept for the `member_type` of `TeamPolicy` and `TeamTask`.
The actual type is defined through the policies, but meets the following API requirements
Note that the specific classes are only part of the public API as provided by `TeamPolicy` and 
`TeamTask` and only their parts as defined by the `TeamHandleConcept`. 
The actual name of the class as well as potential template parameters, existing
constructors, member functions going beyond the concept etc. are NOT part of the public Kokkos API,
and are thus subject to change. 

# Synopsis 
  ```c++
  class TeamHandleConcept {
    public:
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
 
  * ```c++
    TeamHandleConcept()
    TeamHandleConcept(TeamHandleConcept && )
    TeamHandleConcept(const TeamHandleConcept &)
    ~TeamHandleConcept()
    ``` 
    Default, move and copy constructors as well as destructor. 

### Assignment
  * ```c++
    TeamHandleConcept & operator = ( TeamHandleConcept && )
    ```
    Move assignment.
  * ```c++
    TeamHandleConcept & operator = ( const TeamHandleConcept & )
    ```
    Assignment operators.
    Returns: `*this`.

### Index Queries
  * ```c++
    inline int team_size() const noexcept;
    ```
    Returns: the number of threads associated with the team. 
  * ```c++
    inline int team_rank() const noexcept;
    ```
    Returns: the index `i` of the calling thread within the team with `0 <= i < team_size()`
  * ```c++
    inline int league_size() const noexcept;
    ```
    Returns: the number of teams/workitems launched in the kernel. 
  * ```c++
    inline int league_rank() const noexcept;
    ```
    Returns: the index `i` of the calling team within the league with `0 <= i < league_size()`
    
### Scratch Space Control
  * ```c++
    inline const scratch_memory_space & team_scratch(int level) const;
    ```
    This function returns a scratch memory handle shared by all threads in a team, 
    which allows access to scratch memory. This handle can be given as the first 
    argument to a `Kokkos::View` to make it use scratch memory. 
    * `level`: The level of requested scratch memory is either `0` or `1`. 
    Returns: a scratch memory handle to the team shared scratch memory specified by level. 
  * ```c++
    inline const scratch_memory_space & thread_scratch(int level) const;
    ```
    This function returns a scratch memory handle specific to the calling thread, 
    which allows access to its private scratch memory. 
    This handle can be given as the first argument to a `Kokkos::View` to make it use 
    scratch memory. 
    * `level`: The level of requested scratch memory is either `0` or `1`. 
    Returns: a scratch memory handle to the thread scratch memory specified by level. 
  * ```c++
    inline const scratch_memory_space & team_shmem() const;
    ```
    Equivalent to calling `team_scratch(0)`.

### Team Collective Operations
  The following functions must be called collectively by all members of a team. These
  calls must be lexically the same call. I.e. it is not legal to have some members of a team,
  call a collective in one branch and the others in another branch of the code (see example).

  * ```c++
    inline void team_barrier() const noexcept;
    ```
    All members of the team wait at the barrier, until the whole team arrived. This also issues
    a memory fence. 
  * ```c++
    template<class T>
    inline void team_broadcast( T & var , const int source_team_rank ) const noexcept;
    ```
    After this call `var` contains for every member of the team the value of `var` from the thread
    for which `team_rank() == source_team_rank`.   
    * `var`: a variable of type `T` which gets overwritten by the value of `var` from the source rank. 
    * `source_team_rank`: identifies the broadcasting member of the team. 
  * ```c++
    template<class Closure, class T>
    inline void team_broadcast( Closure const & f, T & var , const int source_team_rank ) const noexcept;
    ```
    After this call `var` contains for every member of the team the value of `var` from the thread
    for which `team_rank() == source_team_rank` after applying `f`.
    * `f`: a function object with an `void operator() ( T & )` which is applied to `var` before broadcasting it.
    * `var`: a variable of type `T` which gets overwritten by the value of `f(var)` from the source rank. 
    * `source_team_rank`: identifies the broadcasting member of the team. 
  * ```c++
    template< class ReducerType > 
    inline void team_reduce( ReducerType const & reducer ) const noexcept;
    ```
    Performs a reduction accross all members of the team as specified by `reducer`. 
    `ReducerType` must meet the concept of `Kokkos::Reducer`. 
  * ```c++
    template< class T >
    inline T team_scan( const T & var , T * const global = NULL ) cosnt noexcept;
    ```
    Performs an inclusive scan over the `var` provided by the team members. 
    Let `t = team_rank()` and `VALUES[t]` the value of `var` from thread `t`. 
    Returns: `VALUES[0] + VALUES[1] + `...`+ VALUES[t-1]` or zero for `t==0`.
    * `global` if provided will be set to `VALUES[0] + VALUES[1] + `...`+ VALUES[team_size()-1]`

## Examples

  ```c++
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
  ```

