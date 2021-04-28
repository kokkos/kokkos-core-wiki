
The concept of a `MemorySpace` is the fundamental abstraction to represent the "where" and the "how" that memory allocation and access takes place in Kokkos.  Most code that uses Kokkos should be written to the *generic concept* of a `MemorySpace` rather than any specific instance.  This page talks practically about how to *use* the common features of memory spaces in Kokkos; for a more formal and theoretical treatment, see [this document](Kokkos-Concepts).

> *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates in C++ has used concepts whether they knew it or not.  Please do not be confused by the word "concept" itself, which is now more often associated with a shiny new C++20 language feature.  Here, "concept" just means "what you're allowed to do with a type that is a template parameter in certain places".


## Synopsis

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


## Public Class Members

### Typedefs

  * `memory_space`: The self type;
  * `execution_space`: the default [`ExecutionSpace`](ExecutionSpaceConcept) to use when constructing objects in memory provided by an instance of `MemorySpace`, 
                       or (potentially) when deep copying from or to such memory (see [`deep_copy` documentation](Kokkos%3A%3Adeep_copy) for details). 
                       Kokkos guarantees that `Kokkos::SpaceAccessibility<execution_space, memory_space>::accessible` will be `true` 
                       (see [`Kokkos::SpaceAccessibility`](Kokkos%3A%3ASpaceAccessibility)).
  * `device_type`: `DeviceType<execution_space,memory_space>`.

### Constructors

  * `MemorySpaceConcept()`: Default constructor.
  * `MemorySpaceConcept(const MemorySpaceConcept& src)`: Copy constructor.

### Functions

  * `const char* name() const;`: Returns the label of the memory space instance.
  * `void * allocate(ptrdiff_t size) const;`: Allocates a buffer of at least `size` bytes using the memory resource that `MemorySpaceConcept` represents.
  * `void deallocate(void* ptr, ptrdiff_t size) const;`: Frees the buffer starting at `ptr` (of type `void*`) previously allocated with exactly `allocate(size)`.

## Non Member Facilities

  * `template<class MS> struct is_memory_space;`: typetrait to check whether a class is a memory space.
  * `template<class S1, class S2> struct SpaceAccessibility;`: typetraits to check whether two spaces are compatible (assignable, deep_copy-able, accessable). 


