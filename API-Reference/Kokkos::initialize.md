# Kokkos::initialize

Header File: `Kokkos_Core.hpp`

Usage: 
```c++
   Kokkos::initialize(narg, arg);
   Kokkos::initialize(args);
```

Initialize Kokkos and all enabled Kokkos backends.
This function should be called before calling any other Kokkos API functions,
including Kokkos object constructors.  The function has two overloads.  One takes the same parameters as main() which correspond to the command line parameters for the executable.  The other overload takes a `Kokkos::InitArguments` structure which allows for programmatic control of arguments.

## Interface

```cpp
  Kokkos::initialize(int& narg, char* arg[]);
```

```cpp
  Kokkos::initialize(const InitArguments& args);
```

### Parameters:

  * narg:  number of command line arguments
  * arg: array of command line arguments, valid arguments are listed below.

     * `--kokkos-help`,`--help`: print the valid arguments
     * `--kokkos-threads=INT`,`--threads=INT`: specify total number of threads or number of threads per NUMA region if used in conjunction with the `--numa` option.
     * `--kokkos-numa=INT`,`--numa=INT`: specify number of NUMA regions used by each process. 
     * `--device`,`--device-id`: specify device id to be used by Kokkos (CUDA,HIP,SYCL) 
     * `--num-devices=INT[,INT]`: used when running MPI jobs. Specify number of devices per node to be used. see [Initialization](Initialization) for more detail.

  * args: structure of valid Kokkos arguments

```cpp
struct InitArguments {
  int num_threads;
  int num_numa;
  int device_id;
  int ndevices;
  int skip_device;
  bool disable_warnings;
}
```

    * num_threads: same as `--threads` above
    * num_numa: same as `--numa` above 
    * device_id: same as `--device-id` above
    * ndevices: first argument in `--num-devices` above
    * skip_device: second argument in `--num-devices` above
    * disable_warnings: turn off all Kokkos warnings 

### Requirements

  * `Kokkos::finalize` must be called after `Kokkos::initialize`.
  * `Kokkos::initialize` generally should be called after `MPI_Init` when Kokkos is initialized within an MPI context.
  * User initiated Kokkos objects cannot be constructed until after `Kokkos::initialize` is called.
  * `Kokkos::initialize` may not be called after a call to `Kokkos::finalize`.

### Semantics

  * After calling `Kokkos::initialize`, `Kokkos::is_initialized()` should return true.

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