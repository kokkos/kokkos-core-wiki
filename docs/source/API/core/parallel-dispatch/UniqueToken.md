# `UniqueToken``

Header File: `Kokkos_Core.hpp`

Usage:

```c++
UniqueToken < ExecSpace> token;
UniqueToken < ExecSpace , UniqueTokenScope :: Global > token_foo ;
```

UniqueToken provides consistent, portable identifier for resource allocations and division of work.  The need for UniqueToken arose from the fact thatThread Id is neither consistent nor portable across all execution environments, and within a Functor operator / Lambda, there is no way to identify the active execution resource.  Unique Token use is similar to that of `thread-id`, and has a Scope template parameter (default: `Instance`, but can be `Global`).    


## Interface

```c++
void UniqueToken < ExecSpace > token ;
```

```c++
void UniqueToken < ExecSpace , UniqueTokenScope :: Global > token_foo ;
```

### Parameters

- `ExecutionSpace`:  fundamental abstraction to represent the "where" for Kokkos execution [Kokkos::ExecutionSpaceConcept](../execution_spaces). 
- `UniqueTokenScope`:  defaults to `Instance`, but `Global` can be employed in cases where thread awareness is needed for more than one `ExecutionSpace` instance, as in the submitting concurrent kernels to CUDA streams. 

### Requirements

- `UniqueToken <ExecutionSpace> token` can be called inside a parallel region, but must be released at the end of the iteration.

## Semantics

- In a parallel region, before the main computation, a pool of `UniqueToken` (integer) Id is generated, and each Id is released following iteration.

## Examples

#### Default Usage:

```
UniqueToken < ExecutionSpace > token ;
int number_of_uniqe_ids = token . size ();
RandomGenPool pool ( number_of_unique_ids , seed );
parallel_for ("L", N, KOKKOS_LAMBDA ( int i) {
int id = token . acquire ();
RandomGen gen = pool (id );
...
token . release (id );
});
```
#### `UniqueToken < ExecSpace , UniqueTokenScope :: Global >` token_foo and token_bar provide unique id:

```
void foo () {
UniqueToken < ExecSpace , UniqueTokenScope :: Global > token_foo ;
parallel_for ("L", RangePolicy < ExecSpace >( stream1 ,0,N)
, functor_a ( token_foo ));
}

void bar () {
UniqueToken < ExecSpace , UniqueTokenScope :: Global > token_bar ;
parallel_for ("L", RangePolicy < ExecSpace >( stream2 ,0,N)
, functor_b ( token_bar ));
}

```
