# InitArguments

Defined in `<Kokkos_Core.hpp>` header.

## Interface
```C++
struct InitArguments {  // (deprecated since 3.7)
  int num_threads;
  int num_numa;
  int device_id;
  int ndevices;
  int skip_device;
  bool disable_warnings;
  InitArguments();
};
```

**DEPRECATED: use `Kokkos::InitializationSettings` instead**  
`InitArguments` is a struct that can be used to programmatically define the
arguments passed to [`Kokkos::initialize`](initialize).  It was deprecated in
version 3.7 in favor of
[`Kokkos::InitializationSettings`](InitializationSettings).
One of the main reasons for replacing it was that user-specified data members
cannot be distinguished from defaulted ones.


### Example
```C++
#include <Kokkos_Core.hpp>

int main() {
  Kokkos::InitArguments arguments;
  arguments.num_threads = 2;
  arguments.device_id = 1;
  arguments.disable_warnings = true;
  Kokkos::initialize(arguments);
  // ...
  Kokkos::finalize();
}
```

### See also
* [`Kokkos::InitializationSettings`](InitializationSettings)
* [`Kokkos::initialize`](initialize)
