# Kokkos::Vector

The Kokkos Vector is semantically similar to the std::vector, but it is designed to overcome issues with memory allocations and copies when working with devices that have different memory spaces.  The Kokkos::Vector is a Rank-1 DualView that implements the same interface as the std::vector.  This allows programs that rely heavily on std::vector to grant access to program data from within a non-host execution space.  Note that many of the std::vector compatible functions are host only, so access may be limited based on kernel complexity.  Below is a synopsis of the class and the description for each method specifies whether it is supported on the host, device or both. 

Usage:

```C++
```

## Synopsis
```C++

template <class Scalar, class Arg1Type = void>
class vector : public DualView<Scalar*, LayoutLeft, Arg1Type> {

// Typedefs 

  // Scalar value type
  typedef Scalar value_type;

  // Scalar pointer type
  typedef Scalar* pointer;

  // Const Scalar pointer type
  typedef const Scalar* const_pointer;

  // Scalar reference type
  typedef Scalar& reference;

  // Const Scalar reference type
  typedef const Scalar& const_reference;
  
  // Iterator type
  typedef Scalar* iterator;

  // Const iterator type
  typedef const Scalar* const_iterator;

  // Accessor [Host only]
  KOKKOS_INLINE_FUNCTION reference operator()(int i) const;

  // Accessor [Host only]
  KOKKOS_INLINE_FUNCTION reference operator[](int i) const;

// Constructors

  // Construct empty vector
  vector();

  // Construct vector of size n + 10% and initialize values to `val` 
  vector(int n, Scalar val = Scalar());

  // Resize vector to size n + 10%
  void resize(size_t n);

  // Resize vector to size n + 10% and set values to `val`
  void resize(size_t n, const Scalar& val);

  // Set n values to `val`
  // will auto synchronize between host and device
  void assign(size_t n, const Scalar& val); 

  // same as resize (for compatibility)
  void reserve(size_t n);

  // resize vector to size() + 1 and set last value to val
  // [Host only, auto synchronize device]
  void push_back(Scalar val);

  // reduce size() by 1  
  void pop_back();

  // set size() to 0
  void clear() ;

  // return number of elements in vector
  size_type size() const;

  // return maximum possible number of elements
  size_type max_size() const;

  // return memory used by vector 
  size_type span() const;

  // returns true if vector is empty
  bool empty() const;

  // returns iterator starting at the beginning
  // [Host only]
  iterator begin() const;

  // returns iterator past the last element
  // [Host only]
  iterator end() const;

  // returns reference to the front of the list
  // [Host only]
  reference front();

  // returns reference to the last element in the list
  // [Host only]
  reference back();

  // returns const reference to the front of the list
  // [Host only]
  const_reference front() const;

  // returns const reference to the last element in the list
  // [Host only]
  const_reference back() const;

  // Return the index of largest value satisfying val < comp_val within the
  // range start-theEnd, [Host only]
  size_t lower_bound(const size_t& start, const size_t& theEnd,
                     const Scalar& comp_val) const;
  // Return true if the list is sorted
  bool is_sorted();

  // return iterator pointing to element matching `val` 
  iterator find(Scalar val) const;
  
  // copy data from device to host
  void device_to_host();

  // copy data from host to device
  void host_to_device() const;

  // update/synchronize data in dual view from host perspective
  void on_host();

  // update/synchronize data in dual view from the device perspective 
  void on_device(); 

  // set the data buffer available at the end of the vector
  void set_overallocation(float extra);

};

```