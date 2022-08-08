# Memory Spaces

(CudaSpace)=
## `Kokkos::CudaSpace`

`Kokkos::CudaSpace` is a [`MemorySpace` type](MemorySpaceConcept) representing device memory on a Cuda-capable GPU.  Except in rare instances, it should not be used directly, but instead should be used generically as an memory space.  For details, see [the documentation on the `MemorySpace` concept](MemorySpaceConcept).

(CudaHostPinnedSpace)=
## `Kokkos::CudaHostPinnedSpace`

`Kokkos::CudaHostPinnedSpace` is a [`MemorySpace` type](MemorySpaceConcept) representing host-side pinned memory accessible from a Cuda-capable GPU.  This memory is typically accessible by both host and device execution spaces.  Except in rare instances, it should not be used directly, but instead should be used generically as an memory space.  For details, see [the documentation on the `MemorySpace` concept](MemorySpaceConcept).

(CudaUVMSpace)=
## `Kokkos::CudaUVMSpace`

`Kokkos::CudaUVMSpace` is a [`MemorySpace` type](MemorySpaceConcept) representing unified virtual memory on a Cuda-capable GPU system.  Unified virtual memory is also accessible from most host execution spaces.  Except in rare instances, it should not be used directly, but instead should be used generically as an memory space.  For details, see [the documentation on the `MemorySpace` concept](MemorySpaceConcept).

(HIPSpace)=
## `Kokkos::HIPSpace`

`Kokkos::HIPSpace` <sup>promoted from [Experimental](ExperimentalNamespace) since 4.0</sup> is a [`MemorySpace` type](MemorySpaceConcept) representing device memory on a GPU in the HIP GPU programming environment.  Except in rare instances, it should not be used directly, but instead should be used generically as an memory space.  For details, see [the documentation on the `MemorySpace` concept](MemorySpaceConcept).

(HIPHostPinnedSpace)=
## `Kokkos::HIPHostPinnedSpace`

`Kokkos::HIPHostPinnedSpace` <sup>promoted from [Experimental](ExperimentalNamespace) since 4.0</sup> is a [`MemorySpace` type](MemorySpaceConcept) representing host-side pinned memory accessible from a GPU in the HIP GPU programming environment.  This memory is accessible by both host and device execution spaces.  Except in rare instances, it should not be used directly, but instead should be used generically as an memory space.  For details, see [the documentation on the `MemorySpace` concept](MemorySpaceConcept).

(HIPManagedSpace)=
## `Kokkos::HIPManagedSpace`

`Kokkos::HIPManagedSpace` <sup>promoted from [Experimental](ExperimentalNamespace) since 4.0</sup>  is a [`MemorySpace` type](MemorySpaceConcept) representing page-migrating memory on a GPU in the HIP GPU programming environment.  Page-migrating memory is accessible from most host execution spaces. Even though available with all combinations of operating system and HIP-supported hardware, it requires both operating system and hardware to support and enable the `xnack` feature. Except in rare instances, it should not be used directly, but instead should be used generically as an memory space.  For details, see [the documentation on the `MemorySpace` concept](MemorySpaceConcept).

(HostSpace)=
## `Kokkos::HostSpace`

`Kokkos::HostSpace` is a [`MemorySpace` type](MemorySpaceConcept) representing traditional random access memory accessible from the CPU.  Except in rare instances, it should not be used directly, but instead should be used generically as an memory space.  For details, see [the documentation on the `MemorySpace` concept](MemorySpaceConcept).

(MemorySpaceConcept)=
## `Kokkos::MemorySpaceConcept`

The concept of a `MemorySpace` is the fundamental abstraction to represent the "where" and the "how" that memory allocation and access takes place in Kokkos.  Most code that uses Kokkos should be written to the *generic concept* of a `MemorySpace` rather than any specific instance.  This page talks practically about how to *use* the common features of memory spaces in Kokkos; for a more formal and theoretical treatment, see [this document](KokkosConcepts).

> *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates in C++ has used concepts whether they knew it or not.  Please do not be confused by the word "concept" itself, which is now more often associated with a shiny new C++20 language feature.  Here, "concept" just means "what you're allowed to do with a type that is a template parameter in certain places".

### Synopsis

```c++
// This is not an actual class, it just describes the concept in shorthand
class MemorySpaceConcept {
public: 
  typedef MemorySpaceConcept memory_space;
  typedef ... execution_space;
  typedef Device<execution_space, memory_space> device_type;

  MemorySpaceConcept();
  MemorySpaceConcept(const MemorySpaceConcept& src);
  const char* name() const;
  void * allocate(ptrdiff_t size) const;
  void deallocate(void* ptr, ptrdiff_t size) const;
};

template<class MS>
struct is_memory_space {
enum { value = false };
};

template<>
struct is_memory_space<MemorySpaceConcept> {
enum { value = true };
};
```

### Typedefs

  * `memory_space`: The self type;
  * `execution_space`: the default [`ExecutionSpace`](ExecutionSpaceConcept) to use when constructing objects in memory provided by an instance of `MemorySpace`, 
                       or (potentially) when deep copying from or to such memory (see [`deep_copy` documentation](view/deep_copy) for details). 
                       Kokkos guarantees that `Kokkos::SpaceAccessibility<execution_space, memory_space>::accessible` will be `true` 
                       (see [`Kokkos::SpaceAccessibility`](SpaceAccessibility)).
  * `device_type`: `DeviceType<execution_space,memory_space>`.

### Constructors

  * `MemorySpaceConcept()`: Default constructor.
  * `MemorySpaceConcept(const MemorySpaceConcept& src)`: Copy constructor.

### Functions

  * `const char* name() const;`: Returns the label of the memory space instance.
  * `void * allocate(ptrdiff_t size) const;`: Allocates a buffer of at least `size` bytes using the memory resource that `MemorySpaceConcept` represents.
  * `void deallocate(void* ptr, ptrdiff_t size) const;`: Frees the buffer starting at `ptr` (of type `void*`) previously allocated with exactly `allocate(size)`.

### Non Member Facilities

  * `template<class MS> struct is_memory_space;`: typetrait to check whether a class is a memory space.
  * `template<class S1, class S2> struct SpaceAccessibility;`: typetraits to check whether two spaces are compatible (assignable, deep_copy-able, accessible). 
