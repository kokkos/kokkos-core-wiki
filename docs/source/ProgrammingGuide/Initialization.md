# 5. Initialization

In order to use Kokkos an initialization call is required. That call is responsible for initializing internal objects and acquiring hardware resources such as threads. Typically, this call should be placed right at the start of a program. If you use both MPI and Kokkos, your program should initialize Kokkos right after calling `MPI_Init`. That way, if MPI sets up process binding masks, Kokkos will get that information and use it for best performance. Your program must also _finalize_ Kokkos when done using it in order to free hardware resources.

## 5.0 Include Headers

All primary capabilities of Kokkos are provided by the `Kokkos_Core.hpp` header file.
Some capabilities - specifically data structures in the `containers` subpackage and algorithmic capabilities in the `algorithms` subpackage are included via separate header files.
For specific capabilities check their API reference:
- [API: Core](../API/core-index)
- [API: Containers](../API/containers-index)
- [API: Algorithms](../API/algorithms-index)
- [API in Alphabetical Order](../API/alphabetical)

## 5.1 Initialization by command-line arguments

The simplest way to initialize Kokkos is by calling the following function:
```c++
Kokkos::initialize(int& argc, char* argv[]);
```
Just like `MPI_Init`, this function interprets command-line arguments to determine the requested settings. Also like `MPI_Init`, it reserves the right to remove command-line arguments from the input list. This is why it takes `argc` by reference, rather than by value; it may change the value on output.

During initialization one or more execution spaces will be initialized and assigned to one of the following aliases.

```c++
Kokkos::DefaultExecutionSpace;
```
```c++
Kokkos::DefaultHostExecutionSpace;
```

`DefaultExecutionSpace` is the execution space used with policies and views where one is not explicitly specified.  Primarily, Kokkos will initialize one of the heterogeneous backends (CUDA, OpenMPTarget, HIP, SYCL) as the `DefaultExecutionSpace` if enabled in the build configuration.  In addition, Kokkos requires a `DefaultHostExecutionSpace`.  The `DefaultHostExecutionSpace` is default execution space used when host operations are required.  If one of the parallel host execution spaces are enabled in the build environment then `Kokkos::Serial` is only initialized if it is explicitly enabled in the build configuration.  If a parallel host execution space is not enabled in the build configuration, then `Kokkos::Serial` is initialized as the `DefaultHostExecutionSpace`.
Kokkos chooses the two spaces using the following list:

1. `Kokkos::Cuda`
2. `Kokkos::Experimental::OpenMPTarget`
3. `Kokkos::Experimental::HIP`
4. `Kokkos::Experimental::SYCL`
5. `Kokkos::OpenMP`
6. `Kokkos::Threads`
7. `Kokkos::Experimental::HPX`
8. `Kokkos::Serial`

The highest execution space in the list which is enabled is Kokkos' default execution space, and the highest enabled host execution space is Kokkos' default host execution space. For example, if  `Kokkos::Cuda`, `Kokkos::OpenMP`, and `Kokkos::Serial` are enabled, then `Kokkos::Cuda` is the default execution space and `Kokkos::OpenMP` is the default host execution space.<sup>1</sup>  In cases where the highest enabled backend is a host parallel execution space the `DefaultExecutionSpace` and the `DefaultHostExecutionSpace` will be the same.

Table 5.1 gives a full list of command-line options.

<h4>Table 5.1: Command-line Core options for Kokkos::initialize</h4>

Argument | Description
:---      | :---
  --kokkos-help                  | print this message
  --kokkos-disable-warnings      | disable kokkos warning messages
  --kokkos-print-configuration   | print configuration
  --kokkos-tune-internals        | allow Kokkos to autotune policies and declare tuning features through the tuning system. If left off, Kokkos uses heuristics
  --kokkos-num-threads=INT       | specify total number of threads to use for parallel regions on the host.
  --kokkos-device-id=INT         | specify device id to be used by Kokkos.
  --kokkos-map-device-id-by=(random\|mpi\_rank)| strategy to select device-id automatically from available devices. </br> - random:   choose a random device from available. </br> - mpi_rank: choose device-id based on a round robin assignment of local MPI ranks. Works with OpenMPI, MVAPICH, SLURM, and derived implementations.

You can alternatively set the corresponding environment variable of a flag (all letters in upper-case and underscores instead of hyphens). For example, to disable warning messages, you can either specify `--kokkos-disable-warnings` or set the `KOKKOS_DISABLE_WARNINGS` environment variable to `yes`.

***
<sup>1</sup> This is the preferred set of defaults when CUDA and OpenMP are enabled. If you use a thread-parallel host execution space, we prefer Kokkos' OpenMP back-end, as this ensures compatibility of Kokkos' threads with the application's direct use of OpenMP threads. Kokkos cannot promise that its Threads back-end will not conflict with the application's direct use of operating system threads.

## 5.2 Initialization by struct

Instead of giving [`Kokkos::initialize()`](../API/core/initialize_finalize/initialize) command-line arguments, one may directly pass in initialization parameters using the following struct:

```c++
struct Kokkos::InitArguments {
  int num_threads;
  int num_numa;
  int device_id;
  int ndevices;
  int skip_device;
  bool disable_warnings;
};
```
The `num_threads` field corresponds to the `--kokkos-threads` command-line argument, `num_numa` to `--kokkos-numa`, `device_id` to `--device-id`, `ndevices` to the first value in `--num-devices` and `skip_devices` to the second value in `--num-devices`. (See Table 5.1 for details.) Not all parameters are observed by all execution spaces, and the struct might expand in the future if needed.

If you set `num_threads` or `num_numa` to zero or less, Kokkos will try to determine default values if possible or otherwise set them to 1. In particular, Kokkos can use the `hwloc` library to determine default settings using the assumption that the process binding mask is unique, i.e., that this process does not share any cores with another process. Note that the default value of each parameter is -1.

Here is an example of how to use the struct.

```c++
Kokkos::InitArguments args;
// 8 (CPU) threads per NUMA region
args.num_threads = 8;
// 2 (CPU) NUMA regions per process
args.num_numa = 2;
// If Kokkos was built with CUDA enabled, use the GPU with device ID 1.
args.device_id = 1;

Kokkos::initialize(args);
```

## 5.3 Finalization

At the end of each program, Kokkos needs to be shut down in order to free resources; do this by calling [`Kokkos::finalize()`](../API/core/initialize_finalize/finalize). You may wish to set this to be called automatically at program exit, either by setting an `atexit` hook or by attaching the function to `MPI_COMM_SELF` so that it is called automatically at `MPI_Finalize`.

## 5.4 Example Code

A minimal Kokkos code thus would look like this:

```c++
#include<Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);

  Kokkos::finalize();
}
```
