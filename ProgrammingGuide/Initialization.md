# Chapter 5

# Initialization

In order to use Kokkos an initialization call is required. That call is responsible for acquiring hardware resources such as threads. Typically, this call should be placed right at the start of a program. If you use both MPI and Kokkos, your program should initialize Kokkos right after calling `MPI_Init`. That way, if MPI sets up process binding masks, Kokkos will get that information and use it for best performance. Your program must also _finalize_ Kokkos when done using it in order to free hardware resources.

## 5.1 Initialization by command-line arguments

The simplest way to initialize Kokkos is by calling the following function:
```c++
Kokkos::initialize(int& argc, char* argv[]); 
```
Just like `MPI_Init`, this function interprets command-line arguments to determine the requested settings. Also like `MPI_Init`, it reserves the right to remove command-line arguments from the input list. This is why it takes `argc` by reference, rather than by value; it may change the value on output.

This function will initialize the default execution space
```c++
Kokkos::DefaultExecutionSpace;
```
and its default host execution space
```c++
Kokkos::DefaultHostExecutionSpace;
```
if applicable. It will also always initialize `Kokkos::Serial`. 
These defaults depend on the Kokkos configuration. Kokkos chooses the two spaces using the following list, ordered from low to high:

1. `Kokkos::Threads`
2. `Kokkos::OpenMP`
3. `Kokkos::Cuda`
4. `Kokkos::ROCm`

The highest execution space in the list which is actually enabled is Kokkos' default execution space, and the highest enabled host execution space is Kokkos' default host execution space. For example, if  `Kokkos::Cuda`, `Kokkos::OpenMP`, and `Kokkos::Serial` are enabled, then `Kokkos::Cuda` is the default execution space and `Kokkos::OpenMP` is the default host execution space.<sup>1</sup>

Command-line arguments come in "prefixed" and "non-prefixed" versions. Prefixed versions start with the string `--kokkos-`. `Kokkos::initialize` will remove prefixed options from the input list, but will preserve non-prefixed options. Argument options are given with an equals (`=`) sign. If the same argument occurs more than once, the last one counts. Furthermore, prefixed versions of the command line arguments take precedence over the non-prefixed ones. For example, the arguments

    --kokkos-threads=4 --threads=2

set the number of threads to 4, while

    --kokkos-threads=4 --threads=2 --kokkos-threads=3

set the number of threads to 3. Table 5.1 gives a full list of command-line options.



<h4>Table 5.1: Command-line options for Kokkos::initialize <\h4>

Argument | Description
:---      | :---
--kokkos-help     | print this message
--kokkos-threads  | specify total number of threads or number of threads per NUMA region if used in conjunction with `--numa` option.
--kokkos-numa=INT | specify number of NUMA regions used by process. 
--kokkos-device=INT | specify device id to be used by Kokkos. 
--kokkos-ndevices=INT[,INT] | used when running MPI jobs. Specify number of devices per node to be used. Process to device mapping happens by obtaining the local MPI rank and assigning devices round-robin. The optional second argument allows for an existing device to be ignored. This is most useful on workstations with multiple GPUs, one of which is used to drive screen output.


***
<sup>1</sup> This is the preferred set of defaults when CUDA and OpenMP are enabled. If you use a thread-parallel host execution space, we prefer Kokkos' OpenMP back-end, as this ensures compatibility of Kokkos' threads with the application's direct use of OpenMP threads. Kokkos cannot promise that its Threads back-end will not conflict with the application's direct use of operating system threads.

***


## 5.2 Initialization by struct

Instead of giving `Kokkos::initialize()` command-line arguments, one may directly pass in initialization parameters using the following struct:

```c++
struct Kokkos::InitArguments {
  int num_threads;
  int num_numa;
  int device_id;
  // ... the struct may have more members ...
};
```
The `num_threads` field corresponds to the `--kokkos-threads` command-line argument, `num_numa` to `--kokkos-numa`, and `device_id` to `--kokkos-device`. (See Table 5.1 for details.) Not all parameters are observed by all execution spaces, and the struct might expand in the future if needed.

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

At the end of each program, Kokkos needs to be shut down in order to free resources; do this by calling `Kokkos::finalize()`. You may wish to set this to be called automatically at program exit, either by setting an `atexit` hook or by attaching the function to `MPI_COMM_SELF` so that it is called automatically at `MPI_Finalize`.

**[[Chapter 6: View|View]]**