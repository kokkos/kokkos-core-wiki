The concept of an `ExecutionSpace` is the fundamental abstraction to represent the "where" and the "how" that execution takes place in Kokkos.  Most code that uses Kokkos should be written to the *generic concept* of an `ExecutionSpace` rather than any specific instance.  This page talks practically about how to *use* the common features of execution spaces in Kokkos; for a more formal and theoretical treatment, see [this document](Kokkos-Concepts).

> *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates in C++ has used concepts whether they knew it or not.  Please do not be confused by the word "concept" itself, which is now more often associated with a shiny new C++20 language feature.  Here, "concept" just means "what you're allowed to do with a type that is a template parameter in certain places".

Very Simplest Use: Not at all?
==============================

When first starting to use Kokkos, the (surprising) answer to where you'll see `ExecutionSpace`s used explicitly is "nowhere".  Many of the first things most users learn are "shortcuts" for "do this thing using the default execution space," which is a type alias (a.k.a., `typedef`) named `Kokkos::DefaultExecutionSpace` defined based on build system flags. For instance,

```c++
Kokkos::parallel_for(
  42,
  KOKKOS_LAMBDA (int n) { /* ... */ }
);
```

is a "shortcut" for

```c++
Kokkos::parallel_for(
  Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
    Kokkos::DefaultExecutionSpace(), 0, 42
  ),
  KOKKOS_LAMBDA(int n) { /* ... */ }
);
```

Being more generic
==================

For more intermediate and advanced users, however, it is often good practice to write code that is explicitly generic over the execution space, so that calling code can pass in a non-default execution space if needed.  For instance, if the simple version of your function is

```c++
void my_function(Kokkos::View<double*> data, double scale) {
  Kokkos::parallel_for(
    data.extent(0),
    KOKKOS_LAMBDA (int n) {
      data(n) *= scale;
    }
  );
}
```

then a more advanced, more flexible version of your function might look like:

```c++
template <class ExecSpace, class ViewType>
void my_function(
  ExecSpace ex,
  ViewType data,
  double scale
) {
  static_assert(
    Kokkos::SpaceAccessibility<ExecSpace, typename ViewType::memory_space>::assignable,
    "Incompatible ViewType and ExecutionSpace"
  );
  Kokkos::parallel_for(
    Kokkos::RangePolicy<ExecSpace>(ex, 0, data.extent(0)),
    KOKKOS_LAMBDA (int n) {
      data(n) *= scale;
    }
  );
}
```

More advanced users may also prefer the more explicit form simply to avoid the additional mental exercise of translating "shortcuts" when reading the code later.  Being explicit about *where* and *how* Kokkos parallel patterns are executing tends to reduce bugs, even if it is more verbose.

Functionality
=============

All `ExecutionSpace` types expose a common set of functionality.  In generic code that uses Kokkos (which is pretty much all user code), you should never use any part of an execution space type that isn't common to all execution space types (otherwise, you risk losing portability of your code).  There are a few expressions guaranteed to be valid for any `ExecutionSpace` type.  Given a type `Ex` that is an `ExecutionSpace` type, and an instance of that type `ex`, Kokkos guarantees the following expressions will provide the specified functionality:

---

```c++
ex.name();
```

*Returns:* a value convertible to `const char*` that is guaranteed to be unique to a given `ExecutionSpace` instance type.
*Note:* the pointer returned by this function may not be accessible from the `ExecutionSpace` itself (for instance, on a device); use with caution.

---

```c++
ex.in_parallel();
```

*Returns:* a value convertible to `bool` indicating whether or not the caller is executing as part of a Kokkos parallel pattern.
*Note:* as currently implemented, there is no guarantee that `true` means the caller is necessarily executing as part of a pattern on the particular instance `ex`; just *some* instance of `Ex`.  This may be strengthened in the future.

---

```c++
ex.fence();
```

*Effects:* Upon return, all parallel patterns executed on the instance `ex` are guaranteed to have completed, and their effects are guaranteed visible to the calling thread.
*Returns:* Nothing.
*Note:* This *cannot* be called from within a parallel pattern.  Doing so will lead to unspecified effects (i.e., it might work, but only for some execution spaces, so be extra careful not to do it).

---

```c++
ex.print_configuration(ostr);
ex.print_configuration(ostr, detail);
```

where `ostr` is a `std::ostream` (like `std::cout`, for instance) and `detail` is a boolean indicating whether or not a detailed description should be printed.

*Effects:* Outputs the configuration of `ex` to the given `std::ostream`.
*Returns:* Nothing.
*Note:* This *cannot* be called from within a parallel pattern. 

---

Additionally, the following type aliases (a.k.a. `typedef`s) will be defined by all execution space types:

* `Ex::memory_space`: the default [`MemorySpace`](MemorySpaceConcept) to use when executing with `Ex`.  Kokkos guarantees that `Kokkos::SpaceAccessibility<Ex, Ex::memory_space>::accessible` will be `true` (see [`Kokkos::SpaceAccessibility`](Kokkos%3A%3ASpaceAccessibility))
* `Ex::array_layout`: the default [`ArrayLayout`](ArrayLayoutConcept) recommended for use with `View` types accessed from `Ex`.
* `Ex::scratch_memory_space`: the [`ScratchMemorySpace`](ScratchMemorySpaceConcept) that parallel patterns will use for allocation of scratch memory (for instance, as requested by a [`Kokkos::TeamPolicy`](Kokkos%3A%3ATeamPolicy)).

Default Constructibility, Copy Constructibility
-----------------------------------------------

In addition to the above functionality, all `ExecutionSpace` types in Kokkos are default constructible (you can construct them as `Ex ex()`) and copy constructible (you can construct them as `Ex ex2(ex1)`).  All default constructible instances of an `ExecutionSpace` type are guaranteed to have equivalent behavior, and all copy constructed instances are guaranteed to have equivalent behavior to the instance they were copied from.