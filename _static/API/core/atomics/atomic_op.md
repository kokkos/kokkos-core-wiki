# `atomic_[op]`

Header File: `Kokkos_Core.hpp`

Usage:
```c++
atomic_[op](ptr_to_value,update_value);
```

Atomically updates the `value` at the address given by `ptr_to_value` with `update_value` according to the relevant operation.

## Synopsis

```c++
template<class T>
void atomic_add(T* const ptr_to_value, const T value);

template<class T>
void atomic_and(T* const ptr_to_value, const T value);

template<class T>
void atomic_assign(T* const ptr_to_value, const T value);

template<class T>
void atomic_decrement(T* const ptr_to_value);

template<class T>
void atomic_incrememt(T* const ptr_to_value);

template <class T>
void atomic_max(T* const ptr_to_value, const T value);

template <class T>
void atomic_min(T* const ptr_to_value, const T value);

template<class T>
void atomic_or(T* const ptr_to_value, const T value);

template<class T>
void atomic_sub(T* const ptr_to_value, const T value);
```

## Description

* ```c++
  template<class T>
  void atomic_add(T* const ptr_to_value, const T value);
  ```

  Atomically executes `*ptr_to_value += value`. 
  * `ptr_to_value`: address of the to be updated value.
  * `value`: value to be added.

* ```c++
  template<class T>
  void atomic_and(T* const ptr_to_value, const T value);
  ```

  Atomically executes `*ptr_to_value &= value`. 
  * `ptr_to_value`: address of the to be updated value.
  * `value`: value with which to combine the original value. 

* ```c++
  template<class T>
  void atomic_assign(T* const ptr_to_value, const T value);
  ```

  Atomically executes `*ptr_to_value = value`. 
  * `ptr_to_value`: address of the to be updated value.
  * `value`: new value.

* ```c++
  template<class T>
  void atomic_decrement(T* const ptr_to_value);
  ```

  Atomically executes `(*ptr_to_value)--` or calls `atomic_fetch_sub(ptr_to_value, T(-1))`. 
  * `ptr_to_value`: address of the to be updated value.

* ```c++
  template<class T>
  void atomic_increment(T* const ptr_to_value);
  ```

  Atomically executes `(*ptr_to_value)++` or calls `atomic_fetch_add(ptr_to_value, T(1))`.
  * `ptr_to_value`: address of the to be updated value.

* ```c++
  template<class T>
  void atomic_max(T* const ptr_to_value, const T value);
  ```

  Atomically executes `if (value > *ptr_to_value) *ptr_to_value = value`. 
  * `ptr_to_value`: address of the to be updated value.
  * `value`: value which to take the maximum with.

* ```c++
  template<class T>
  void atomic_min(T* const ptr_to_value, const T value);
  ```

  Atomically executes `if (value < *ptr_to_value) *ptr_to_value = value`. 
  * `ptr_to_value`: address of the to be updated value.
  * `value`: value which to take the minimum with.

* ```c++
  template<class T>
  void atomic_or(T* const ptr_to_value, const T value);
  ```

  Atomically executes `*ptr_to_value |= value`. 
  * `ptr_to_value`: address of the to be updated value.
  * `value`: value with which to combine the original value. 

* ```c++
  template<class T>
  void atomic_sub(T* const ptr_to_value, const T value);
  ```

  Atomically executes `*ptr_to_value -= value`. 
  * `ptr_to_value`: address of the to be updated value.
  * `value`: value to be subtracted.
