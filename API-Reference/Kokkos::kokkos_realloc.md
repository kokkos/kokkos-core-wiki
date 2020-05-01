# `Kokkos::kokkos_realloc`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void* kokkos_realloc(void* ptr, size_t new_size);
```

Reallocates the given area of memory. It must be previously allocated by `Kokkos::kokkos_malloc()` or `Kokkos::realloc()` and not yet freed with `Kokkos::kokkos_free()`, otherwise, the results are undefined.

## Parameters

* `ptr`: The pointer to the memory area to be reallocated.
* `new_size`: The new size in bytes.

## Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the default memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`).

## Return value

(none)
