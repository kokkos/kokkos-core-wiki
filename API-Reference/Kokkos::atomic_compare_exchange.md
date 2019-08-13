# `Kokkos::atomic_exchange`

Header File: `Kokkos_Core.hpp`

Usage:
  ```c++
  old_val = atomic_compare_exchange(ptr_to_value,comparison_value, new_value);
  ```

Atomicly sets the value at the address given by `ptr_to_value` to `new_value` if the current value at `ptr_to_value`
is equal to `comparison_value`, and returns the previously stored value at the address independent on whether 
the exchange has happened.

## Synopsis

```c++
  template<class T>
  T atomic_compare_exchange(T* const ptr_to_value, const T new_value);
```

## Description

* ```c++
  template<class T>
  T atomic_compare_exchange(T* const ptr_to_value, const T comparison_value, const T new_value);
  ```

  Atomicly executes `old_value = *ptr_to_value; &ptr_to_value = new_value; return old_value;`. 
  * `ptr_to_value`: address of the to be updated value.
  * `new_value`: new value.
  * `old_value`: value at address `ptr_to_value` before doing the exchange.


