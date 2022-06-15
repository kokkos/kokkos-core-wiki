# ExecutionPolicy

The concept of an `ExecutionPolicy` is the fundamental abstraction to represent "how" the execution of a Kokkos parallel pattern takes place.  This page talks practically about how to *use* the common features of execution policies in Kokkos; for a more formal and theoretical treatment, see [this document](Kokkos-Concepts).

> *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates in C++ has used concepts whether they knew it or not.  Please do not be confused by the word "concept" itself, which is now more often associated with a shiny new C++20 language feature.  Here, "concept" just means "what you're allowed to do with a type that is a template parameter in certain places".

## What is an `ExecutionPolicy`?

The dominant parallel dispatch mechanism in Kokkos, described [elsewhere in the programming guide](ParallelDispatch), involves a `parallel_pattern` (e.g., something like `Kokkos::parallel_for` or `Kokkos::parallel_reduce`), an `ExecutionPolicy`, and a `Functor`.  In a hand-wavy sense:

```c++
parallel_pattern(
  ExecutionPolicy(),
  Functor()
);
```

The most basic ("beginner") case is actually a shortcut:

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

In this example, `Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>` is the `ExecutionPolicy` type.

### Functionality

All `ExecutionPolicy` types provide a nested type named `index_type`.
