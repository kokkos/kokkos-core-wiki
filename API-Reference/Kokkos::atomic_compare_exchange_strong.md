# `Kokkos::atomic_compare_exchange_strong`

Header File: `Kokkos_Core.hpp`

Usage:
  ```c++
  was_exchanged = atomic_compare_exchange_strong(ptr_to_value,comparison_value, new_value);
  ```

Atomically sets the value at the address given by `ptr_to_value` to `new_value` if the current value at `ptr_to_value`
is equal to `comparison_value`, and returns true if the exchange has happened.

## Synopsis

```c++
  template<class T>
  bool atomic_compare_exchange_strong(T* const ptr_to_value, const T comparison_value, const T new_value);
```

## Description

* ```c++
  template<class T>
  bool atomic_compare_exchange_strong(T* const ptr_to_value, const T comparison_value, const T new_value);
  ```

  Atomically executes `old_value = *ptr_to_value; if(old_value==comparison_value) *ptr_to_value = new_value; return old_value;`. 
  * `ptr_to_value`: address of the to be updated value.
  * `comparison_value`: value to be comapred to.
  * `new_value`: new value.
  * `old_value`: value at address `ptr_to_value` before doing the exchange.


