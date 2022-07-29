# finalize

Header File: `Kokkos_Core.hpp`

Shut down all enabled Kokkos backends and free all associated resources.
This function should be called after calling all other Kokkos API functions,
*including Kokkos object destructors*.

Programs are ill-formed if they do not call this function after calling [`Kokkos::initialize`](initialize).

Usage: 
```c++
Kokkos::finalize();
```

## Interface

```c++
Kokkos::finalize();
```

### Parameters:

   * None

### Requirements
   * `Kokkos::finalize` must be called before `MPI_Finalize` if Kokkos is used in an MPI context.
   * `Kokkos::finalize` must be called after user initialized Kokkos objects are out of scope. 

### Semantics

   * `Kokkos::is_initialized()` should return false after calling `Kokkos::finalize`

### Example

```c++
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);

  // add scoping to ensure my_view destructor is called before Kokkos::finalize  
  {
     Kokkos::View<double*> my_view("my_view", 10);
  }
 
  Kokkos::finalize();
  
}
```
