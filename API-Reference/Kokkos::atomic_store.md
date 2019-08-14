# `Kokkos::atomic_exchange`

Header File: `Kokkos_Core.hpp`

Usage:
  ```c++
  atomic_exchange(ptr_to_value,new_value);
  ```

Atomicly sets the value at the address given by `ptr_to_value` to `new_value`.

## Synopsis

```c++
  template<class T>
  void atomic_exchange(T* const ptr_to_value, const T new_value);
```

## Description

* ```c++
  template<class T>
  void atomic_exchange(T* const ptr_to_value, const T new_value);
  ```

  Atomicly executes `*ptr_to_value = new_value;`. 
  * `ptr_to_value`: address of the to be updated value.
  * `new_value`: new value.


