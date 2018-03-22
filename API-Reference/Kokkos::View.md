# Kokkos::View

Header File: `Kokkos_Core.hpp`

Usage: 

Kokkos View is a potentially reference counted multi dimensional array with compile time layouts and memory space.
Its semantics are similar to that of `std::shared_ptr`. 

## Interface

```cpp
template <class DataType [, class LayoutType] [, class MemorySpace] [, class MemoryTraits]>
class View;
```

### Parameters:


### Requriements:
  

## Public Class Members

### Constructors

  * `View()`: Default Constructor. No allocations are made, no reference counting happens. All extents are zero and its data pointer is NULL.
  * `View(const std::string& name, const IntType& n0, ... , const IntType& nR)`: Standard allocating constructor
    * `name` is a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique,
    * 

### Data Access Functions

  * `value_type& operator() (const IntType& i0, ..., const IntType& iR) const`

### Property Introspection


## Examples

More Detailed Examples are provided in the ExecutionPolicy documentation. 

```c++
#include<Kokkos_Core.hpp>
#include<cstdio> 

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc,argv);

   int N = atoi(argv[1]);

   Kokkos::parallel_for("Loop1", N, KOKKOS_LAMBDA (const int& i) {
     printf("Greeting from iteration %i\n",i);
   });

   Kokkos::finalize();
}
```

```c++
#include<Kokkos_Core.hpp>
#include<cstdio> 

struct TagA {};
struct TagB {};

struct Foo {
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagA, const Kokkos::TeamPolicy<>::member_type& team) const {
    printf("Greetings from thread %i of team %i with TagA\n",team.thread_rank(),team.league_rank());
  });
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagB, const Kokkos::TeamPolicy<>::member_type& team) const {
    printf("Greetings from thread %i of team %i with TagB\n",team.thread_rank(),team.league_rank());
  });
});

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc,argv);

   int N = atoi(argv[1]);

   Foo foo;

   Kokkos::parallel_for(Kokkos::TeamPolicy<Tag1>(N,Kokkos::AUTO), foo);
   Kokkos::parallel_for("Loop2", Kokkos::TeamPolicy<Tag2>(N,Kokkos::AUTO), foo);
   
   Kokkos::finalize();
}
```


