# C-style memory management

(kokkos_malloc)=
## `Kokkos::kokkos_malloc`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void* kokkos_malloc(const std:string& label, size_t size);
```

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void* kokkos_malloc(size_t size);
```

Allocate `size` bytes of uninitialized storage on the specified memory space [`MemorySpace`](memory_spaces) plus some extra space for metadata such as the label.

If allocation succeeds, returns a pointer to the lowest (first) byte in the allocated memory block that is suitably aligned for any scalar type.

If allocation fails, an exception of type `Kokkos::Experimental::RawMemoryAllocationFailure` is thrown.

### Parameters

`label`: A user provided string which is used in profiling and debugging tools via the KokkosP Profiling Tools.  
`size`: The number of bytes to allocate.

### Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`).

### Return value

On success, returns the pointer to the beginning of newly allocated memory.
To avoid a memory leak, the returned pointer must be deallocated with [`Kokkos::kokkos_free()`](kokkos_free) or [`Kokkos::realloc()`](kokkos_realloc).

### Exceptions

On failure, throws `Kokkos::Experimental::RawMemoryAllocationFailure`.

<br/>

(kokkos_realloc)=
## `Kokkos::kokkos_realloc`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void* kokkos_realloc(void* ptr, size_t new_size);
```

Reallocates the given area of memory. It must be previously allocated by [`Kokkos::kokkos_malloc()`](kokkos_malloc) or [`Kokkos::kokkos_realloc()`](kokkos_realloc) on the same memory space [`MemorySpace`](memory_spaces) and not yet freed with [`Kokkos::kokkos_free()`](kokkos_free), otherwise, the results are undefined.

### Parameters

`ptr`: The pointer to the memory area to be reallocated.  
`new_size`: The new size in bytes.

### Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`).

### Return value

On success, returns a pointer to the beginning of the newly allocated memory. To avoid a memory leak, the returned pointer must be deallocated with [`Kokkos::kokkos_free()`](kokkos_free), the original pointer `ptr` is invalidated and any access to it is undefined behavior (even if reallocation was in-place).

On failure, returns a null pointer. The original pointer ptr remains valid and may need to be deallocated with [`Kokkos::kokkos_free()`](kokkos_free).

### Exceptions

On failure, throws `Kokkos::Experimental::RawMemoryAllocationFailure`.

<br/>

(kokkos_free)=
## `Kokkos::kokkos_free`

Defined in header `<Kokkos_Core.hpp>`

```c++
template <class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
void kokkos_free(void* ptr);
```

Deallocates the space previously allocated by [`Kokkos::kokkos_malloc()`](kokkos_malloc) or [`Kokkos::kokkos_realloc()`](kokkos_realloc) on the specified memory space `MemorySpace`.

If `ptr` is a null pointer, the function does nothing.

### Parameters

`ptr`: The pointer to the memory to deallocate on the specified memory space.

### Template parameters

* `MemorySpace`:  Controls the storage location. If omitted the memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`).

### Return value

(none)
### Exceptions

Throws `std::runtime_error` on failure to deallocate.
