# `kokkos_free`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
void kokkos_free(void* ptr);
```

Deallocates the space previously allocated by [`Kokkos::kokkos_malloc()`](./malloc) or [`Kokkos::kokkos_realloc()`](./realloc) on the specified memory space `MemorySpace`.

If `ptr` is a null pointer, the function does nothing.

## Parameters

`ptr`: The pointer to the memory to deallocate on the specified memory space.

## Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`).

## Return value

(none)
## Exceptions

Throws `std::runtime_error` on failure to deallocate.

