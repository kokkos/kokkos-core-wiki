# `Kokkos::atomic_exchange`

Header File: `Kokkos_Core.hpp`

Usage:
  ```c++
  old_val = atomic_exchange(ptr_to_value,new_value);
  ```

Atomically sets the value at the address given by `ptr_to_value` to `new_value` and returns the previously stored value at the address.

## Synopsis

```c++
  template<class T>
  T atomic_exchange(T* const ptr_to_value, const T new_value);
```

## Description

* ```c++
  template<class T>
  T atomic_exchange(T* const ptr_to_value, const T new_value);
  ```

  Atomically executes `old_value = *ptr_to_value; *ptr_to_value = new_value; return old_value;`. 
  * `ptr_to_value`: address of the to be updated value.
  * `new_value`: new value.
  * `old_value`: value at address `ptr_to_value` before doing the exchange.


