
.. role:: cppkokkos(code)
	:language: cppkokkos

``UnorderedMap``
================

Header file: ``<Kokkos_UnorderedMap.hpp>``

Kokkos's unordered map is designed to efficiently handle tens of thousands of concurrent insertions.
Consequently, the API is significantly different from the standard unordered_map.
The two key differences are *fixed capacity* and *index based*.

- *Fixed capacity*: The capacity of the unordered_map is fixed when inside a parallel algorithm.
  This means that an insert can fail when the capacity of the map is exceeded.
  The capacity of the map can be changed (rehash) from the host.

- *Index based*: Instead of returning pointers or iterators (which would not work when moving
  between memory spaces) the map uses integer indexes. This also allows the map to store data
  in cache friendly ways. The availability of indexes is managed by an internal atomic bitset based on ``uint32_t``.


Description
-----------

.. cppkokkos:class:: template <typename Key, typename Value, typename Device = Kokkos::DefaultExecutionSpace> UnorderedMap

   :tparam Key: Must be a POD (Plain Old Data type)

   :tparam Value: `void` indicates an unordered set, otherwise must be trivially copyable

   :tparam Device: Device is any class or struct with the following public typedefs or type aliases: `execution_space`, `memory_space`, and `device_type`

   .. rubric:: Constructor

   .. cppkokkos:function:: UnorderedMap(uint32_t capacity_hint);

      Create map with enough space for at least capacity_hint number of objects

      .. warning:: Host Only

   .. rubric:: Public Member Functions

   .. cppkokkos:function:: clear();

      Clear the map

      .. warning:: Host Only

   .. cppkokkos:function:: bool rehash(uint32_t requested_capacity);

      Rehash map to given capacity, the current size is used as a lower bound O(capacity)

      .. warning:: Host Only

   .. cppkokkos:function:: uint32_t size() const;

      Current size of the map, O(capacity)

      .. warning:: Host Only

   .. cppkokkos:kokkosinlinefunction:: uint32_t capacity() const;

       Capacity of the map, O(1)

   .. cppkokkos:kokkosinlinefunction:: UnorderedMapInsertResult insert(key) const;

      Insert the given key into the map with a default constructed value

   .. cppkokkos:kokkosinlinefunction:: UnorderedMapInsertResult insert(Key key, Value value, Insert op = NoOp) const;

      Insert the given key/value pair into the map and optionally specify
      the operator, op, used for combining values if key already exists

   .. cppkokkos:kokkosinlinefunction:: uint32_t find(Key key) const

      Return the index of the key if it exist, otherwise return invalid_index

   .. cppkokkos:kokkosinlinefunction:: bool exists(Key key) const;

      Does the key exist in the map

   .. cppkokkos:kokkosinlinefunction:: bool valid_at(uint32_t index) const;

      Is the current index a valid key/value pair

   .. cppkokkos:kokkosinlinefunction:: Key key_at(uint32_t index) const;

      Return the current key at the index

   .. cppkokkos:kokkosinlinefunction:: Value value_at(uint32_t index) const;

      Return the current value at the index

   .. cppkokkos:kokkosinlinefunction:: constexpr bool is_allocated() const;

      Return true if the internal views (keys, values, hashmap) are allocated


.. cppkokkos:class:: UnorderedMapInsertResult

   .. rubric:: Public Methods

   .. cppkokkos:kokkosinlinefunction:: bool success() const;

      Was the key/value pair successfully inserted into the map

   .. cppkokkos:kokkosinlinefunction:: bool existing() const;

      Is the key already present in the map

   .. cppkokkos:kokkosinlinefunction:: bool failed() const;

      Did the insert fail?

   .. cppkokkos:kokkosinlinefunction:: uint32_t index() const;

      Index where the key exists in the map as long as failed() == false

.. cppkokkos:struct:: template <class ValueTypeView, class ValuesIdxType> UnorderedMapInsertOpTypes

   :tparam ValueTypeView: The UnorderedMap value array type.

   :tparam ValuesIdxType: The index type for lookups in the value array.

   .. rubric:: *Public* Insertion Operator Types

   .. cppkokkos:struct:: NoOp

        Insert the given key/value pair into the map

   .. cppkokkos:struct:: AtomicAdd

       Duplicate key insertions sum values together.


.. _unordered_map_insert_op_types_noop:

Insertion using default ``UnorderedMapInsertOpTypes::NoOp``
-----------------------------------------------------------

There are 3 potential states for every insertion which are reported by the ``UnorderedMapInsertResult``:

- ``success``: implies that the current thread has successfully inserted its key/value pair

- ``existing``: implies that the key is already in the map and its current value is unchanged

- ``failed`` means that either the capacity of the map was exhausted or that a free index was not found
  with a bounded search of the internal atomic bitset. A ``failed`` insertion requires the user to increase
  the capacity (``rehash``) and restart the algoritm.

.. code-block:: cpp

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
  
Insertion using ``UnorderedMapInsertOpTypes::AtomicAdd``
--------------------------------------------------------

The behavior from :ref:`unordered_map_insert_op_types_noop` holds true with the
exception that the ``UnorderedMapInsertResult``:

- ``existing`` implies that the key is already in the map and the existing value at key was summed
  with the new value being inserted.

.. code-block:: cpp

    // use the AtomicAdd insert operation
    using map_op_type     = Kokkos::UnorderedMapInsertOpTypes<value_view_type, size_type>;
    using atomic_add_type = typename map_op_type::AtomicAdd;
    atomic_add_type atomic_add;
    parallel_for(N, KOKKOS_LAMBDA (uint32_t i) {
      map.insert(i, values(i), atomic_add);
    });


Iteration
---------

Iterating over Kokkos' ``UnorderedMap`` is different from iterating over a standard container. The pattern is to iterate over the capacity of the map and check if the current index is valid.

Example
~~~~~~~

.. code-block:: cpp

    // assume umap is an existing Kokkos::UnorderedMap
    parallel_for(umap.capacity(), KOKKOS_LAMBDA (uint32_t i) {
        if( umap.valid_at(i) ) {
            auto key   = umap.key_at(i);
            auto value = umap.value_at(i);
            ...
        }
    });
