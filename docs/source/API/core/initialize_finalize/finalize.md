# finalize

Defined in header `<Kokkos_Core.hpp>`

Usage: 
```C++
Kokkos::finalize();
```

Terminates the Kokkos execution environment.
This functions cleans up all Kokkos states and released the associated
resources.
Once this function is called, no Kokkos API functions (not even
[`Kokkos::initialize`](initialize)) may be called, except for
`Kokkos::is_initialized` or `Kokkos::is_finalized`.
The user must ensure that all Kokkos objects (e.g. `Kokkos::View`) are detroyed
before `Kokkos::finalize` before `Kokkos::finalize` gets called.

Programs are ill-formed if they do not call this function after calling [`Kokkos::initialize`](initialize).

## Interface

```C++
Kokkos::finalize();
```

### Requirements
   * `Kokkos::finalize` must be called before `MPI_Finalize` if Kokkos is used in an MPI context.
   * `Kokkos::finalize` must be called after user initialized Kokkos objects are out of scope. 

### Semantics

   * `Kokkos::is_initialized()` should return false after calling `Kokkos::finalize`

### Example

```C++
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {  // scope to ensure that my_view destructor is called before Kokkos::finalize
     Kokkos::View<double*> my_view("my_view", 10);
  }  // scope of my_view ends here
  Kokkos::finalize();
}
```
