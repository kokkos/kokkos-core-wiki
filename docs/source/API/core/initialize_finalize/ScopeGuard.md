# ScopeGuard

Header File: `Kokkos_Core.hpp`

Usage: 
```c++
Kokkos::ScopeGuard guard(argc, argv);
Kokkos::ScopeGuard guard(Kokkos::InitializationSettings()  // (since 3.7)
                             .set_map_device_id_by("random")
                             .set_num_threads(1));
```

`ScopeGuard` is a class which ensure thats [`Kokkos::initialize`](initialize) and
[`Kokkos::finalize`](finalize) are called correctly even in the presence of unhandled
exceptions.
It calls [`Kokkos::intialize`](initialize) with the provided arguments in the
constructor and [`Kokkos::finalize`](finalize) in the destructor.


**WARNING: change of behavior in version 3.7**  
Since Kokkos version 3.7, `ScopeGuard` unconditionally forwards the provided
arguments to `Kokkos::initialize`, which means they have the same precondition.
Until version 3.7, `ScopeGuard` was calling `Kokkos::initialize` in its
constructor only if `Kokkos::is_initialized()` was `false`, and it was calling
`Kokkos::finalize` in its destructor only if it called `Kokkos::initialize` in
its constructor.

We dropped support for the old behavior.  If you think you really need it, you
may do:
```C++
auto guard = std::unique_ptr<Kokkos::ScopeGuard>(
    Kokkos::is_initialized() ? new Kokkos::ScopeGuard() : nullptr);
```
or
```C++
auto guard = std::optional<Kokkos::ScopeGuard>(
    Kokkos::is_initialized() ? Kokkos::ScopeGuard() : std::nullopt);
```
with C++17.  This will work regardless of the Kokkos version.

## Interface

```c++
class ScopeGuard {
public:
  ScopeGuard(ScopeGuard const&) = delete;
  ScopeGuard(ScopeGuard&&) = delete;
  ScopeGuard& operator=(ScopeGuard const&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;

  ScopeGuard(int& argc, char* argv[]);                           // (until 3.7)
  ScopeGuard(InitArguments const& arguments = InitArguments());  // (until 3.7)

  template <class... Args>
  ScopeGuard(Args&&... args) {                                   // (since 3.7)
    // possible implementation
    initialize(std::forward<Args>(args)...);
  }

  ~ScopeGuard() {
    // possible implementation
    finalize();
  }
```

### Parameters:

* `argc`: number of command line arguments
* `argv`: array of character pointers to null-terminated strings storing the command line arguments
* `arguments`: `struct` object with valid initialization arguments
* `args`: arguments to pass to [`Kokkos::initialize`](initialize)

Note that all of the parameters above are passed to the `Kokkos::initialize` called internally.  See [Kokkos::initialize](initialize) for more details.

### Example

```c++
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  Kokkos::View<double*> my_view("my_view", 10);
  // my_view destructor called before Kokkos::finalize
  // ScopeGuard destructor called, calls Kokkos::finalize
}
```

### See also
* [Kokkos::initialize](initialize)
* [Kokkos::finalize](finalize)
