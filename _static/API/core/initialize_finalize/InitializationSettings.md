# InitializationSettings

Defined in header `<KokkosCore.cpp>`

Usage:
```C++
auto settings = Kokkos::InitializationSettings()
                    .set_num_threads(8)
                    .set_device_id(0)
                    .set_disable_warnings(false);
```

`InitializationSettings` is a class that can be used to define the settings for
initializating Kokkos programmatically without having to call the two parameter
form (`argc` and `argv`) of [`Kokkos::initialize()`](./initialize).
It was introduced in version 3.7 as a replacement for the
`Kokkos::InitArguments` structure.

## Interface

```C++
class InitializationSettings {
public:
  InitializationSettings();
  InitializationSettings(InitArguments const& arguments);  // deprecated
  InitializationSettings& set_PARAMETER-NAME(PARAMETER-TYPE value);  // see below
  bool has_PARAMETER-NAME() const;  // see below
  PARAMETER-TYPE get_PARAMETER-NAME() const;  // see below
};
```

The table below summarizes what settings are available.
*PARAMETER-NAME* | *PARAMETER-TYPE* | Description
--- | --- | ---
`num_threads` | `int` | Number of threads to use with the host parallel backend.  Must be greater than zero.
`device_id` | `int` | Device to use with the device parallel backend.  Valid IDs are zero to number of GPU(s) available for execution minus one.
`map_device_id_by` | `std::string` | Strategy to select a device automatically from the GPUs available for execution. Must be either `"mpi_rank"` for round-robin assignment based on the local MPI rank or `"random"`.
`disable_warnings` | `bool` | Whether to disable warning messages.
`print_configuration` | `bool` | Whether to print the configuration after initialization.
`tune_internals` | `bool` | Whether to allow autotuning internals instead of using heuristics.
`tools_libs` | `std::string` | Which tool dynamic library to load. Must either be the full path to library or the name of library if the path is present in the runtime library search path (e.g. `LD_LIBRARY_PATH`)
`tools_help` | `bool` | Query the loaded tool for its command-line options support.
`tools_args` | `std::string` | Options to pass to the loaded tool as command-line arguments.


```C++
InitializationSettings();
```
Constructs a new object that does not contain any value for any of the settings.

```C++
InitializationSettings(InitArgument const& arguments);  // deprecated
```
Converts the deprecated structure to a new object.
Data members from the structure that compare equal to their default value are
assumed to be unset.

Let `PARAMETER-NAME` be a valid setting of type `PARAMETER-TYPE` as defined in the table above.

```C++
InitializationSettings& set_PARAMETER-NAME(PARAMETER-NAME value);
```
Replaces the content of the `PARAMETER-NAME` setting with `value` and return a
reference to the object.
`value` must be a valid value for `PARAMETER-NAME`.

```C++
bool has_PARAMETER-NAME() const;
```
Checks whether the object contains a value for the `PARAMETER-NAME` setting.
Returns `true` if it contains a value, `false` otherwise.

```C++
PARAMETER-TYPE get_PARAMETER-NAME() const;
```
Accesses the contained value for the `PARAMETER-NAME` setting.
The behavior is undefined if the object does not contain a value for setting
`PARAMETER-NAME`.

### Example

```c++
#include <Kokkos_Core.hpp>

int main() {
  Kokkos::initialize(Kokkos::InitializationSettings()
                         .set_print_configuration(true)
                         .set_map_device_id_by("random")
                         .set_num_threads(1));
  // ...
  Kokkos::finalize();
}
```

### See also
* [`Kokkos::initialize`](./initialize): initializes the Kokkos execution environment
