# `UnorderedMap`

Kokkos's unordered map is designed to efficiently handle tens of thousands of concurrent insertions.  Consequently, the API is significantly different from the standard unordered_map.  The two key differences are *fixed capacity* and *index based*.

*Fixed capacity*:  The capacity of the unordered_map is fix when inside a parallel algorithm.  This means that an insert can fail when the capacity of the map is exceeded.  The capacity of the map can be changed (rehash) from the host.

*Index based*:  Instead of returning pointers or iterators (which would not work when moving between memory spaces) the map uses integer indexes.  This also allows the map to store data in cache friendly ways.  The availability of indexes is managed by an internal atomic bitset based on `uint32_t`.

An `UnorderedMap` behaves like an unordered set if the template parameter `Value` is void.

## Kokkos::UnorderedMap API

```c++
class UnorderedMapInsertResult;
struct UnorderedMapInsertOpTypes;

template< typename Key    // Must be a POD
        , typename Value  // void indicates an unordered set, otherwise 
                          // must be trivially copyable
        , typename Device = Kokkos::DefaultExecutionSpace
                          // Device is any class or struct with the 
                          // following typedefs or type aliases
                          // execution_space, memory_space, and device_type
        >
class UnorderedMap {
public:
  // Host: create map with enough space for at least
  // capacity_hint number of objects
  UnorderedMap(uint32_t capacity_hint);
  
  // Host: clear the map
  void clear();
  
  // Host: rehash map to given capacity,
  // the current size is used as a lower bound
  // O(capacity)
  bool rehash(uint32_t requested_capacity);
  
  // Host: current size of the map, O(capacity)
  uint32_t size() const;
  
  // Host/Device: capacity of the map, O(1)
  uint32_t capacity() const;
  
  // Device: insert the given key into the map with a 
  // default constructed value
  UnorderedMapInsertResult insert(key) const;
  
  // Device: insert the given key/value pair into the map
  // Optionally specify the operator, op, used for combining
  // values if a key already exists.
  UnorderedMapInsertResult insert(Key key, Value value, InsertOp op = NoOp) const;
  
  // Device: return the index of the key if it exist, otherwise 
  // return invalid_index
  uint32_t find(Key key) const
  
  // Device: Does the key exist in the map 
  bool exists(Key key) const

  // Device: is the current index a valid key/value pair
  bool valid_at(uint32_t index) const;
  
  // Device: return the current key at the index
  Key key_at(uint32_t index) const;
  
  // Device: return the current value at the index
  Value value_at(uint32_t index) const;

  // Host/Device: return true if the internal views (keys, values, 
  // hashmap) are allocated
  constexpr bool is_allocated() const;
 
};


class UnorderedMapInsertResult {
public:
  // Was the key/value pair successfully inserted into the map
  bool success() const;
  
  // Is the key already present in the map
  bool existing() const;
  
  // Did the insert fail?
  bool failed() const;
  
  // Index where the key exists in the map
  // as long as failed() == false
  uint32_t index() const;
};

template <  class ValueTypeView // The UnorderedMap value array type.
          , class ValuesIdxType // The index type for lookups in the value array.
         >
struct UnorderedMapInsertOpTypes { 
  // NoOp (default): the first key inserted stores the associated value.
  struct NoOp;
  
  // AtomicAdd: duplicate key insertions sum values together.
  struct AtomicAdd;
  };
};
```

## Insertion using the default UnorderedMapInsertOpTypes::Noop

There are 3 potential states for every insertion which are reported by the `UnorderedMapInsertResult`: success, existing, and failed.  `success` implies that the current thread has successfully inserted its key/value pair.  `existing` implies that the key is already in the map and its current value is unchanged.  `failed` means that either the capacity of the map was exhausted or that a free index was not found with a bounded search of the internal atomic bitset.  A `failed` insertion requires the user to increase the capacity (`rehash`) and restart the algoritm.

```c++
// use the default NoOp insert operation
using map_op_type = Kokkos::UnorderedMapInsertOpTypes<value_view_type, size_type>;
using noop_type   = typename map_op_type::NoOp;
noop_type noop;
parallel_for(N, KOKKOS_LAMBDA (uint32_t i) {
  map.insert(i, values(i), noop);
});
// OR;
parallel_for(N, KOKKOS_LAMBDA (uint32_t i) {
  map.insert(i, values(i));
});
```

## Insertion using the default UnorderedMapInsertOpTypes::AtomicAdd

The behavior from [above](Insertion using the default UnorderedMapInsertOpTypes::Noop) hold true with the exception that the `UnorderedMapInsertResult`: `existing` implies that the key is already in the map and the existing value at key was summed with the new value being inserted.

```c++
// use the AtomicAdd insert operation
using map_op_type = Kokkos::UnorderedMapInsertOpTypes<value_view_type, size_type>;
using atomic_add_type   = typename map_op_type::AtomicAdd;
atomic_add_type atomic_add;
parallel_for(N, KOKKOS_LAMBDA (uint32_t i) {
  map.insert(i, values(i), atomic_add);
});
```

## Iteration

Iterating over Kokkos' `UnorderedMap` is different from iterating over a standard container.  The pattern is to iterate over the capacity of the map and check if the current index is valid.

Example:

```c++
// assume umap is an existing Kokkos::UnorderedMap
parallel_for(umap.capacity(), KOKKOS_LAMBDA (uint32_t i) {
  if( umap.valid_at(i) ) {
    auto key   = umap.key_at(i);
    auto value = umap.value_at(i);
    ...
  }
});
```
