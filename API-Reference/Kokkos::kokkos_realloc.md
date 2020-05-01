# `Kokkos::kokkos_realloc`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void* kokkos_realloc(void* ptr, size_t new_size);
```

Reallocates the given area of memory. It must be previously allocated by [`Kokkos::kokkos_malloc()`](Kokkos%3A%3Akokkos_malloc) or [`Kokkos::kokkos_realloc()`](Kokkos%3A%3Akokkos_realloc) and not yet freed with [`Kokkos::kokkos_free()`](Kokkos%3A%3Akokkos_free), otherwise, the results are undefined.

## Parameters

`ptr`: The pointer to the memory area to be reallocated.  
`new_size`: The new size in bytes.

## Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`).

## Return value

On success, returns a pointer to the beginning of the newly allocated memory. To avoid a memory leak, the returned pointer must be deallocated with [`Kokkos::kokkos_free()`](Kokkos%3A%3Akokkos_free), the original pointer `ptr` is invalidated and any access to it is undefined behavior (even if reallocation was in-place).

On failure, returns a null pointer. The original pointer ptr remains valid and may need to be deallocated with [`Kokkos::kokkos_free()`](Kokkos%3A%3Akokkos_free).

## Exceptions

On failure, throws `Kokkos::Experimental::RawMemoryAllocationFailure`.
