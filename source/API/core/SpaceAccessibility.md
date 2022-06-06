
# Space Accessibility

`Kokkos::SpaceAccessibility<>` is a traits class template that takes an [`ExecutionSpace` type](ExecutionSpaceConcept) or [`MemorySpace` type](MemorySpaceConcept) as the first template argument and a `MemorySpace` type as the second type and expresses details about the relationship between those entities.  Given memory space types `MSp1` and `MSp2` and an execution space type `Ex`, the following expressions will be valid with the specified meaning:

---

```c++
Kokkos::SpaceAccessibility<Ex, MSp1>::accessible
```

A compile-time value convertible to `bool` guaranteed to be `true` if and only if accessing memory allocated by an instance of `MSp1` from a thread of execution provided by an instance of `Ex` (e.g., within a parallel pattern launched on an instance of `Ex`) is well-defined and valid within the Kokkos programming model for *all* instances of `Ex` and *all* instances of `MSp1`.

---

```c++
Kokkos::SpaceAccessibility<MSp1, MSp2>::accessible
```

Equivalent to `Kokkos::SpaceAccessibility<MSp1::execution_space, MSp2>::accessible`.

---

```c++
Kokkos::SpaceAccessibility<MSp1, MSp2>::assignable
```

A compile-time value convertible to `bool` guaranteed to be `true` if and only if it is valid within the Kokkos programming model to assign values from  any (otherwise valid) instance of `Kokkos::View` type `V2` (with `std::is_same<V2::memory_space, MSp2>::value` equal to `true`) to references retrieved from any (otherwise valid) instance of a `View` type `V1` (with `std::is_same<V1::memory_space, MSp1>::value` equal to `true`).

---

```c++
Kokkos::SpaceAccessibility<Ex, MSp1>::assignable
```

Equivalent to `Kokkos::SpaceAccessibility<Ex::memory_space, MSp1>::assignable`.


---

```c++
Kokkos::SpaceAccessibility<MSp1, MSp2>::deepcopy
```

A compile-time value convertible to `bool` guaranteed to be `true` if and only if it is valid within the Kokkos programming model to [`Kokkos::deep_copy`](Kokkos%3A%3Adeep_copy) from any (otherwise valid) instance of `Kokkos::View` type `V2` (with `std::is_same<V2::memory_space, MSp2>::value` equal to `true`) to any (otherwise valid and otherwise compatible) instance of a `View` type `V1` (with `std::is_same<V1::memory_space, MSp1>::value` equal to `true`).  In other words, if `v2` is a valid instance of `V2` and `v1` is a valid instance of `V1` (with shape and other attributes otherwise compatible with `v2`), the following expression will be well-defined and valid in the Kokkos programming model:

```c++
Kokkos::deep_copy(v1, v2);
```

---

```c++
Kokkos::SpaceAccessibility<Ex, MSp1>::deepcopy
```

Equivalent to `Kokkos::SpaceAccessibility<Ex::memory_space, MSp1>::deepcopy`.


---


Additionally, the following nested type names will be defined:


```c++
Kokkos::SpaceAccessibility<Ex, MSp1>::space
```

An "intercessory" memory space that should be used to deep copy memory for access by any instance of `Ex`.  Formally, a type meeting the [requirements of `Kokkos::Device`](Kokkos%3A%3ADevice) with the following expressions all `true` at compile-time:

- `Kokkos::SpaceAccessibility<Ex, Kokkos::SpaceAccessibility<Ex, MSp1>::space::memory_space>::accessible`
- `Kokkos::SpaceAccessibility<Kokkos::SpaceAccessibility<Ex, MSp1>::space::memory_space, MSp1>::deepcopy`
- `Kokkos::SpaceAccessibility<Ex, Kokkos::SpaceAccessibility<Ex, MSp1>::space::memory_space>::deepcopy`

---

```c++
Kokkos::SpaceAccessibility<MSp1, MSp2>::space
```

Equivalent to `Kokkos::SpaceAccessibility<MSp1::execution_space, MSp2>::space`.