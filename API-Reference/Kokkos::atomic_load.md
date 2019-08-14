# `Kokkos::atomic_load`

Header File: `Kokkos_Core.hpp`

Usage:
  ```c++
  value = atomic_load(ptr_to_value);
  ```

Atomicly reads the value at the address given by `ptr_to_value`.

## Synopsis

```c++
  template<class T>
  T atomic_load(T* const ptr_to_value);
```

## Description

* ```c++
  template<class T>
  T atomic_load(T* const ptr_to_value);
  ```

  Atomicly executes `value = *ptr_to_value; return value;`. 
  * `ptr_to_value`: address of the to be updated value.
  * `value`: value at address `ptr_to_value`.


