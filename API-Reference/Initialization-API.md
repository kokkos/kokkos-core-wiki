## Kokkos::initialize

```c++
  Kokkos::initialize();
  Kokkos::initialize(const InitArguments&);
  Kokkos::initialize(int& argc, char* argv[]);
```

Initializes all enabled Kokkos backends.
This function should be called before calling any other Kokkos API functions,
including Kokkos object constructors.
`Kokkos::finalize` must be called after `Kokkos::initialize`.

## Kokkos::finalize

```c++
  Kokkos::finalize();
```

Shuts down all enabled Kokkos backends and frees all associated resources.
This function should be called after calling all other Kokkos API functions,
*including Kokkos object destructors*.
Programs are ill-formed if they do not call this function after calling `Kokkos::initialize`.

## Kokkos::ScopeGuard

`ScopeGuard` is a class introduced to ensure that `Kokkos::initialize` and
`Kokkos::finalize` are called correctly even in the presence of unhandled
exceptions and multiple libraries trying to "own" Kokkos initialization.
`ScopeGuard` calls `Kokkos::initialize` in its constructor only if
`Kokkos::is_initialized()` is false, and calls `Kokkos::finalize` in its
destructor only if it called `Kokkos::initialize` in its constructor.

### Kokkos::ScopeGuard::ScopeGuard
```c++
  ScopeGuard();
  ScopeGuard(const InitArguments&);
  ScopeGuard(int& argc, char* argv[]);
```
Calls `Kokkos::initialize` only if
`Kokkos::is_initialized()` is false.
Arguments are passed directly to `Kokkos::initialize` if it is called.

### Kokkos::ScopeGuard::~ScopeGuard
```c++
  ~ScopeGuard();
```
Calls `Kokkos::finalize`
only if the constructor of this object called `Kokkos::initialize`.

One common mistake is allowing Kokkos objects to live past `Kokkos::finalize`,
which is easy to do since all objects in the scope of `main()` live past that point.
Here is an example of this mistake:
```c++
int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  Kokkos::View<double*> my_view("my_view", 10);
  Kokkos::finalize();
  // my_view destructor called after Kokkos::finalize !
}
```
Switching to `Kokkos::ScopeGuard` fixes it:
```c++
int main(int argc, char** argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);
  Kokkos::View<double*> my_view("my_view", 10);
  // my_view destructor called before Kokkos::finalize
  // ScopeGuard destructor called, calls Kokkos::finalize
}
```
