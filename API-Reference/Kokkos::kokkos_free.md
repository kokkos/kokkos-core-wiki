# `Kokkos::kokkos_free`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void kokkos_free(void* ptr);
```

Deallocates the space previously allocated by [`Kokkos::kokkos_malloc`](Kokkos%3A%3Akokkos_malloc) or [`Kokkos::kokkos_realloc`](Kokkos%3A%3Akokkos_realloc).

If `ptr` is a null pointer, the function does nothing.

## Parameters

* `ptr`: The pointer to the memory to deallocate on the specified memory space.

## Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the default memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`).

## Return value

On success, returns a pointer to the beginning of the newly allocated memory. To avoid a memory leak, the returned pointer must be deallocated with `Kokkos::kokkos_free()`, the original pointer `ptr` is invalidated and any access to it is undefined behavior (even if reallocation was in-place).

On failure, returns a null pointer. The original pointer ptr remains valid and may need to be deallocated with `Kokkos::kokkos_free()`.
