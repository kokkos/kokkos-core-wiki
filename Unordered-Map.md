# Unordered Map

Kokkos's unordered map is designed to efficently handle tens of thousands of concurrent insertions.  Consequently, the API is signifcantly different from the standard unordered_map.  The two key differences are *fixed capacity* and *index based*.

*Fixed capacity*:  The capacity of the unordered_map is fix when inside a parallel algorithm.  This means that an insert can fail when the capacity of the map is exceeded.  The capacity of the map can be changed (rehash) from the host.

*Index based*:  Instead of returning pointers or iterators (which would not work when moving between memory spaces) the map uses integer indexes.  This also allows the map to store data in cache friendly ways.  The availablity of indexes is managed by an internal atomic bitset based on `uint32_t`.

An `UnorderedMap` behaves like an unordered set if the template parameter `Value` is void.

## Kokkos::UnorderedMap API

```c++
class UnorderedMapInsertResult;

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
  UnorderedMapInsertResult insert(Key key, Value value) const;
  
  // Device: return the index of the key if it exist, otherwise 
  // return invalid_index
  uint32_t find(Key key) const
  
  // Device: Does the key exist in the map 
  bool exist(Key key) const

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
```

## Insertion

There are 3 potential states for every insertion which are reported by the `UnorderedMapInsertResult`: success, existing, and failed.  `success` implies that the current thread has successfully inserted its key/value pair.  `existing` implies that the key is already in the map and its current value is unchanged.  `failed` means that either the capacity of the map was exhausted or that a free index was not found with a bounded search of the internal atomic bitset.  A `failed` insertion requires the user to increase the capacity (`rehash`) and restart the algoritm.


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




