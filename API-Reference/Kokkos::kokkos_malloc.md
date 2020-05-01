# `Kokkos::kokkos_malloc`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void kokkos_malloc(const std:string& label, size_t size);
```

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void kokkos_malloc(size_t size);
```

Allocate `size` bites of unitialized storage on the specified memory space.

If allocation succeeds, returns a pointer to the lowest (first) byte in the allocated memory block that is suitably aligned for any scalar type.

## Parameters

  * `label`: A user provided string which is used in profiling and debugging tools via the KokkosP Profiling Tools.
  *  `size: The number of bytes to allocate.

## Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the default memory space of the default execution space is used (i.e. Kokkos::DefaultExecutionSpace::memory_space`).

## Return value`

On success, returns the pointer to the beginning of newly allocated memory.
To avoid a memory leak, the returned pointer must be deallocated with `Kokkos::kokkos_free()` or `Kokkos::realloc()`.

On failure, returns a null pointer.
