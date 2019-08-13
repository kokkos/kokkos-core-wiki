The concept of a `MemorySpace` is the fundamental abstraction to represent the "where" and the "how" that memory allocation and access takes place in Kokkos.  Most code that uses Kokkos should be written to the *generic concept* of a `MemorySpace` rather than any specific instance.  This page talks practically about how to *use* the common features of memory spaces in Kokkos; for a more formal and theoretical treatment, see [this document](Kokkos-Concepts).

> *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates in C++ has used concepts whether they knew it or not.  Please do not be confused by the word "concept" itself, which is now more often associated with a shiny new C++20 language feature.  Here, "concept" just means "what you're allowed to do with a type that is a template parameter in certain places".

Very Simplest Use: Not at all?
==============================

Just like the concept of an [`ExecutionSpace`](ExecutionSpaceConcept), the first place you'll see `MemorySpace`s used explicitly is "nowhere".  Many of the first things most users learn are "shortcuts" for "use the default memory space for this" which is a type alias (a.k.a., `typedef`) named `Kokkos::DefaultExecutionSpace::memory_space` defined based on build system flags. For instance,

```c++
auto v1 = Kokkos::View<double*>("v1", 42);
```

is actually short for

```c++
auto v1 =
  Kokkos::View<double*,
    Kokkos::DefaultExecutionSpace::array_layout,
    Kokkos::DefaultExecutionSpace::memory_space
  >("v1", 42);
```

(which, in turn, is short for a slightly longer version of this that uses [`Kokkos::Device<>`](Kokkos%3A%3ADevice)).  As with the `ExecutionSpace` concept, advanced users may wish to be generic with respect to the `MemorySpace` that their function operates on; see the analogous section of [this document](ExecutionSpaceConcept) for more discussion.


Functionality
=============

All `MemorySpace` types expose a common set of functionality.  In generic code that uses Kokkos (which is pretty much all user code), you should never use any part of an memory space type that isn't common to all memory space types (otherwise, you risk losing portability of your code).  There are a few expressions guaranteed to be valid for any `MemorySpace` type.  Given a type `MSp` that is a `MemorySpace` type, and an instance of that type `msp`, Kokkos guarantees the following expressions will provide the specified functionality:

---

```c++
msp.name();
```

*Returns:* a value convertible to `const char*` that is guaranteed to be unique to a given `MemorySpace` instance type.

---

```c++
msp.allocate(size);
```

*Effects:* Allocates a buffer of at least `size` bytes using the memory resource that `msp` represents.
*Returns:* a `void*` pointing to the beginning of a buffer that is at least `size` bytes.


---

```c++
msp.deallocate(ptr, size);
```

*Effects:* Frees the buffer starting at `ptr` (of type `void*`) previously allocated with exactly `msp.allocate(size)`.
*Notes:* This must correspond to a previous `allocate` on some instance with equivalent effects to `msp` (see section on default constructibility and copy constructibility below).  The corresponding allocation must occur in a thread whose effects are visible to the caller of `deallocate` (see the discussion of the `fence()` method of the [`ExecutionSpace` concept](ExecutionSpaceConcept)).


---

Additionally, the following type aliases (a.k.a. `typedef`s) will be defined by all memory space types:

* `MSp::execution_space`: the default [`ExecutionSpace`](ExecutionSpaceConcept) to use when constructing objects in memory provided by an instance of `MSp`, or (potentially) when deep copying from or to such memory (see [`deep_copy` documentation](Kokkos%3A%3Adeep_copy) for details).  Kokkos guarantees that `Kokkos::SpaceAccessibility<MSp::execution_space, MSp>::accessible` will be `true` (see [`Kokkos::SpaceAccessibility`](Kokkos%3A%3ASpaceAccessibility)).

Default Constructibility, Copy Constructibility
-----------------------------------------------

Like `ExecutionSpace`, in addition to the above functionality, all `MemorySpace` types in Kokkos are default constructible (you can construct them as `MSp msp()`) and copy constructible (you can construct them as `MSp msp2(msp1)`).  All default constructible instances of a `MemorySpace` type are guaranteed to have equivalent behavior, and all copy constructed instances are guaranteed to have equivalent behavior to the instance they were copied from.
