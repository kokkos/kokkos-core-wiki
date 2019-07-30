Custom arbitrary reductions are implemented using a reduction class and a "reduced" class.  The "reduced" class is much like the custom scalar type used with [BuiltIn Reducers](Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types) and the reduction class implements the [ReducerConcept](Kokkos%3A%3AReducerConcept)


## Example

This example performs a custom reduction on an array using a custom class and reducer. 

```c++
namespace sample {

template< class ScalarType, int N >
struct array_type {
  ScalarType myArray[N];

  KOKKOS_INLINE_FUNCTION
  array_type() {
     init();
  }

  KOKKOS_INLINE_FUNCTION
  array_type(const array_type & rhs) {
     for (int i = 0; i < N; i++ ){
        myArray[i] = rhs.myArray[i];
     }
  }


  KOKKOS_INLINE_FUNCTION  // initialize myArray to 0
  void init() {
      for (int i = 0; i < N; i++ ) { myArray[i] = 0; }
   }

  KOKKOS_INLINE_FUNCTION
  array_type& operator += (const array_type& src) {
    for ( int i = 0; i < N; i++ ) {
       myArray[i]+=src.myArray[i];
    }
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  void operator += (const volatile array_type& src) volatile {
    for ( int i = 0; i < N; i++ ) {
      myArray[i]+=src.myArray[i];
    }
  }

};

template<class T, class Space, int N>
struct SumMyArray {
public:
  //Required
  typedef SumMyArray reducer;
  typedef array_type<T,N> value_type;
  typedef Kokkos::View<value_type*, Space, Kokkos::MemoryUnmanaged> result_view_type;

private:
  value_type & value;

public:

  KOKKOS_INLINE_FUNCTION
  SumMyArray(value_type& value_): value(value_) {}

  //Required
  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src)  const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dest, const volatile value_type& src) const {
    dest += src;
  }

  KOKKOS_INLINE_FUNCTION
  void init( value_type& val)  const {
    val.init();
  }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const {
    return value;
  }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const {
    return result_view_type(&value);
  }

  KOKKOS_INLINE_FUNCTION
  bool references_scalar() const {
    return true;
  }
};

int main( int argc, char* argv[] ) {

  Kokkos::initialize( argc, argv );
  {
     int E = 1024;

     typedef array_type<int,4> ValueType;
     typedef SumMyArray <int, Kokkos::HostSpace, 4> ArraySumResult;

     ValueType tr;

     Kokkos::parallel_reduce (  E, [&] (const int j, ValueType &upd ) {
        int ndx =i%4;  // sum all of the i%4 entries (divide total by 4)
        upd.myArray[ndx] += 1;
     }, ArraySumResult(tr) );

     // Output result.
     printf( "  Computed result %d, %d, %d, %d \n", 
         tr.myArray[0], tr.myArray[1], tr.myArray[2], tr.myArray[3] );
  }
  Kokkos::finalize();
}

}

```
 