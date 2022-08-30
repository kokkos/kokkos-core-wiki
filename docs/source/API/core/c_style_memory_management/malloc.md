# `kokkos_malloc`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
void* kokkos_malloc(const std:string& label, size_t size);
```

```c++
template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
void* kokkos_malloc(size_t size);
```

Allocate `size` bytes of uninitialized storage on the specified memory space [`MemorySpace`](../memory_spaces) plus some extra space for metadata such as the label.

If allocation succeeds, returns a pointer to the lowest (first) byte in the allocated memory block that is suitably aligned for any scalar type.

If allocation fails, an exception of type `Kokkos::Experimental::RawMemoryAllocationFailure` is thrown.

WARNING: calling any function that manipulates the behavior of the memory (e.g. `memAdvise`) on memory managed by `Kokkos` results in undefined behavior.

## Parameters

`label`: A user provided string which is used in profiling and debugging tools via the KokkosP Profiling Tools.  
`size`: The number of bytes to allocate.

## Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`).

## Return value

On success, returns the pointer to the beginning of newly allocated memory.
To avoid a memory leak, the returned pointer must be deallocated with [`Kokkos::kokkos_free()`](free) or [`Kokkos::realloc()`](realloc).

## Exceptions

On failure, throws `Kokkos::Experimental::RawMemoryAllocationFailure`.

