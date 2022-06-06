# Kokkos::ScopeGuard

Header File: `Kokkos_Core.hpp`

Usage: 
```c++
   Kokkos::ScopeGuard(narg, arg);
   Kokkos::ScopeGuard(args);
```

`ScopeGuard` is a class which ensure that `Kokkos::initialize` and
`Kokkos::finalize` are called correctly even in the presence of unhandled
exceptions and multiple libraries trying to "own" Kokkos initialization.
`ScopeGuard` calls `Kokkos::initialize` in its constructor only if
`Kokkos::is_initialized()` is false, and calls `Kokkos::finalize` in its
destructor only if it called `Kokkos::initialize` in its constructor.

## Interface

```cpp
  Kokkos::ScopeGuard(int& narg, char* arg[]);
```

```cpp
  Kokkos::ScopeGuard(const InitArguments& args);
```

```cpp
  ~ScopeGuard();
```


### Parameters:

  * narg:  number of command line arguments
  * arg: array of command line arguments, valid arguments are listed below.
  * args: structure of valid Kokkos arguments

Note that all of the parameters above are passed to the `Kokkos::initialize` called internally.  See [Kokkos::initialize](Kokkos%3A%3Ainitialize) for more details.
 
### Requirements
  * `Kokkos::ScopeGuard` object should be constructed before user initiated Kokkos objects 

### Semantics
  * Calls `Kokkos::initialize` only if `Kokkos::is_initialized()` is false.
  * Arguments are passed directly to `Kokkos::initialize` if it is called.
  * Kokkos::ScopeGuard::~ScopeGuard calls `Kokkos::finalize` only if the constructor of this object called `Kokkos::initialize`.

### Example


```c++
int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  Kokkos::View<double*> my_view("my_view", 10);
  // my_view destructor called before Kokkos::finalize
  // ScopeGuard destructor called, calls Kokkos::finalize
}
```
