# Kokkos::View

Header File: `Kokkos_Core.hpp`

Usage: 

Kokkos View is a potentially reference counted multi dimensional array with compile time layouts and memory space.
Its semantics are similar to that of `std::shared_ptr`. 

## Interface

```cpp
template <class DataType [, class LayoutType] [, class MemorySpace] [, class MemoryTraits]>
class View;
```

### Parameters

Template parameters other than `DataType` are optional, but ordering is enforced. That means for example that `LayoutType` can be omitted but if both `MemorySpace` and `MemoryTraits` are specified, `MemorySpace` must come before `MemoryTraits`.
  
  * `DataType`: Defines the fundamental scalar type of the `View` and its dimensionality. The basic structure is `ScalarType STARS BRACKETS` where the number of STARS denotes the number of runtime length dimensions and the number of BRACKETS defines the compile time dimensions. Due to C++ type restrictions runtime dimensions must come first. Examples:
    * `double**`: 2D View of `double` with 2 runtime dimensions
    * `const int***[5][3]`: 5D View of `int` with 3 runtime and 2 compile dimensions. The data is `const`.
    * `Foo[6][2]`: 2D View of a class `Foo` with 2 compile time dimensions.
  * `LayoutType`: determines the mapping of indices into the underlying 1D memory storage. Custom Layouts can be implemented, but Kokkos comes with some built-in ones: 
    * `LayoutRight`: strides increase from the right most to the left most dimension. The last dimension has a stride of one. This corresponds to how C multi dimensional arrays (`[][][]`) are laid out in memory. 
    * `LayoutLeft`: strides increase from the left most to the right most dimension. The first dimension has a stride of one. This is the layout Fortran uses for its arrays. 
    * `LayoutStride`: strides can be arbitrary for each dimension. 
  * `MemorySpace`: Controls the storage location. If omitted the default memory space of the default execution space is used (i.e. `Kokkos::DefaultExecutionSpace::memory_space`)
  * `MemoryTraits`: Sets access properties via enum parameters for the templated `Kokkos::MemoryTraits<>` class. Enums can be combined bit combined. Posible values:
    * `Unmanaged`: The View will not be reference counted. The allocation has to be provided to the constructor.
    * `Atomic`: All accesses to the view will use atomic operations. 
    * `RandomAccess`: Hint that the view is used in a random access manner. If the view is also `const` this will trigger special load operations on GPUs (i.e. texture fetches).
    * `Restrict`: There is no aliasing of the view by other data structures in the current scope. 

### Requirements:
  

## Public Class Members

### Enums

 * `rank`: rank of the view (i.e. the dimensionality).
 * `rank_dynamic`: number of runtime determined dimensions.
 * `reference_type_is_lvalue_reference`: whether the reference type is a C++ lvalue reference. 

### Typedefs

#### Data Types

 *  `data_type`: The `DataType` of the View, note `data_type` contains the array specifiers (e.g. `int**[3]`)
 *  `const_data_type`: Const version of `DataType`, same as `data_type` if that is already const.
 *  `non_const_data_type`: Non-const version of `DataType`, same as `data_type` if that is already non-const.

 *  `scalar_array_type`: If `DataType` represents some properly specialised array data type such as Sacado FAD types, `scalar_array_type` is the underlying fundamental scalar type.
 *  `const_scalar_array_type`: Const version of `scalar_array_type`, same as `scalar_array_type` if that is already const
 *  `non_const_scalar_array_type`: Non-Const version of `scalar_array_type`, same as `scalar_array_type` if that is already non-const.

#### Scalar Types
 *  `value_type`: The `data_type` stripped of its array specifiers, i.e. the scalar type of the data the view is referencing (e.g. if `data_type` is `const int**[3]`, `value_type` is `const int`.
 *  `const_value_type`: const version of `value_type`.
 *  `non_const_value_type`: non-const version of `value_type`.
  


#### Spaces
 *  `execution_space`: Execution Space associated with the view, will be used for performing view initialization, and certain deep_copy operations.
 *  `memory_space`: Data storage location type. 
 *  `device_type`: the compound type defined by `Device<execution_space,memory_space>`
 *  `memory_traits`: The memory traits of the view. 
 *  `host_mirror_space`: Host accessible memory space used in `HostMirror`.



#### ViewTypes
 * `non_const_type`: this view type with all template parameters explicitly defined.
 * `const_type`: this view type with all template parameters explicitly defined using a `const` data type.
 * `HostMirror`: compatible view type with the same `DataType` and `LayoutType` stored in host accessible memory space. 

#### Data Handles
 * `reference_type`: return type of the view access operators.
 * `pointer_type`: pointer to scalar type. 

#### Other
 *  `array_layout`: The Layout of the View.
 *  `size_type`: index type associated with the memory space of this view. 
 *  `dimension`: An integer array like type, able to represent the extents of the view.
 *  `specialize`: A specialization tag used for partial specialization of the mapping construct underlying a Kokkos View.

### Constructors

  * `View()`: Default Constructor. No allocations are made, no reference counting happens. All extents are zero and its data pointer is NULL.
  * `View( const View<DT, Prop...>& rhs)`: Copy constructor with compatible view. Follows View assignment rules. 
  * `View( View&& rhs)`: Move constructor
  * `View( const std::string& name, const IntType& ... indices)`: Standard allocating constructor.
    * `name`: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique,
    * `indices`: Runtime dimensions of the view.  
    * Requires: `sizeof(IntType...)==rank_dynamic()` 
    * Requires: `array_layout::is_regular == true`.
  * `View( const std::string& name, const array_layout& layout)`: Standard allocating constructor.  
    * `name`: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique,
    * `layout`: an instance of a layout class.
  * `View( const AllocProperties& prop, , const IntType& ... indices)`: Allocating constructor with allocation properties.
    * An allocation properties object is returned by the `view_alloc` function. 
    * `indices`: Runtime dimensions of the view.
    * Requires: `sizeof(IntType...)==rank_dynamic()` 
    * Requires: `array_layout::is_regular == true`.
  * `View( const AllocProperties& prop, const array_layout& layout)`: Allocating constructor with allocation properties and a layout object. 
    * An allocation properties object is returned by the `view_alloc` function. 
    * `layout`: an instance of a layout class.
  * `View( const pointer_type& ptr, const IntType& ... indices)`: Unmanaged data wrapping constructor.
    * `ptr`: pointer to a user provided memory allocation. Must provide storage of size `View::required_allocation_size(n0,...,nR)`
    * `indices`: Runtime dimensions of the view.   
    * Requires: `sizeof(IntType...)==rank_dynamic()` 
    * Requires: `array_layout::is_regular == true`.
  * `View( const std::string& name, const array_layout& layout)`: Unmanaged data wrapper constructor.  
    * `ptr`: pointer to a user provided memory allocation. Must provide storage of size `View::required_allocation_size(layout)` (*NEEDS TO BE IMPLEMENTED*)
    * `layout`: an instance of a layout class.
  * `View( const ScratchSpace& space, const IntType& ... indices)`: Constructor which acquires memory from a Scratch Memory handle.
    * `space`: scratch memory handle. Typically returned from `team_handles` in `TeamPolicy` kernels. 
    * `indices`: Runtime dimensions of the view.   
    * Requires: `sizeof(IntType...)==rank_dynamic()` 
    * Requires: `array_layout::is_regular == true`.
  * `View( const ScratchSpace& space, const array_layout& layout)`: Constructor which acquires memory from a Scratch Memory handle.  
    * `space`: scratch memory handle. Typically returned from `team_handles` in `TeamPolicy` kernels. 
    * `layout`: an instance of a layout class.
  * `View( const View<DT, Prop...>& rhs, Args ... args)`: Subview constructor. See `subview` function for arguments. 
 
### Data Access Functions

  * ```c++
    reference_type operator() (const IntType& ... indices) const
    ```
    Returns a value of `reference_type` which may or not be referenceable itself. The number of index arguments must match the `rank` of the view.
    See notes on `reference_type` for properties of the return type. 
    * Requires: `sizeof(IntType...)==rank_dynamic()` 

  * ```c++
    reference_type access (const IntType& i0=0, ... , const IntType& i7=0) const
    ```
    Returns a value of `reference_type` which may or not be referenceable itself. The number of index arguments must be equal or larger than the `rank` of the view.
    Index arguments beyond `rank` must be `0`, which will be enforced if `KOKKOS_DEBUG` is defined. 
    See notes on `reference_type` for properties of the return type. 

### Data Layout, Dimensions, Strides

  * ```c++
    constexpr array_layout layout() const
    ```
    Returns the layout object. Can be used to to construct other views with the same dimensions.  
  * ```c++
    template<class iType>
    constexpr size_t extent( const iType& dim) const
    ```
    Return the extent of the specified dimension. `iType` must be an integral type, and `dim` must be smaller than `rank`.
  * ```c++
    template<class iType>
    constexpr int extent_int( const iType& dim) const
    ```
    Return the extent of the specified dimension as an `int`. `iType` must be an integral type, and `dim` must be smaller than `rank`.
    Compared to `extent` this function can be useful on architectures where `int` operations are more efficient than `size_t`. It also may eliminate the need for type casts in applications which otherwise perform all index operations with `int`. 
  * ```c++
    template<class iType>
    constexpr size_t stride(const iType& dim) const
    ```
    Return the stride of the specified dimension. `iType` must be an integral type, and `dim` must be smaller than `rank`.
    Example: `a.stride(3) == (&a(i0,i1,i2,i3+1,i4)-&a(i0,i1,i2,i3,i4))`
  * ```c++
    constexpr size_t stride_0() const
    ```
    Return the stride of dimension 0. 
  * ```c++
    constexpr size_t stride_1() const
    ```
    Return the stride of dimension 1. 
  * ```c++
    constexpr size_t stride_2() const
    ```
    Return the stride of dimension 2. 
  * ```c++
    constexpr size_t stride_3() const
    ```
    Return the stride of dimension 3. 
  * ```c++
    constexpr size_t stride_4() const
    ```
    Return the stride of dimension 4. 
  * ```c++
    constexpr size_t stride_5() const
    ```
    Return the stride of dimension 5. 
  * ```c++
    constexpr size_t stride_6() const
    ```
    Return the stride of dimension 6. 
  * ```c++
    constexpr size_t stride_7() const
    ```
    Return the stride of dimension 7. 
  * ```c++
    constexpr size_t span() const
    ```
    Returns the memory span in elements between the element with the lowest and the highest address. This can be larger than the product of extents due to padding, and or non-contiguous data layout as for example `LayoutStride` allows. 
  * ```c++
    constexpr size_t size() const
    ```
    Returns the product of extents, i.e. the logical number of elements in the view. 
  * ```c++
    constexpr pointer_type data() const
    ```
    Return the pointer to the underlying data allocation.
  * ```c++
    bool span_is_contiguous() const
    ```
    Whether the span is contiguous (i.e. whether every memory location between in span belongs to the index space covered by the view).
  * ```c++
    static constexpr size_t required_allocation_size(size_t N0 = 0, ..., size_t N8 = 0);
    ```
    Returns the number of bytes necessary for an unmanaged view of the provided dimensions. This function is only valid if `array_layout::is_regular == true`.
  * ```c++
    static constexpr size_t required_allocation_size(const array_layout& layout);
    ```
    Returns the number of bytes necessary for an unmanaged view of the provided layout.
  
### Other
  * ```c++
    int use_count() const;
    ```
    Returns the current reference count of the underlying allocation.

  * ```c++
    const char* label() const;
    ```
    Returns the label of the View. 

  * ```c++
    const bool is_assignable(const View<DT, Prop...>& rhs);
    ```
    Returns true if the View can be assigned to rhs.  See below for assignment rules. 

  * ```c++
    void assign_data(pointer_type arg_data);
    ```
    Decrement reference count of previously assigned data and set the underlying pointer to arg_data.  Note that the effective result of this operation is that the view is now an unmanaged view; thus, the deallocation of memory associated with arg_data is not linked in anyway to the deallocation of the view. 

  * ```c++
    constexpr bool is_allocated() const;
    ```
    Returns true if the view points to a valid memory location.  This function works for both managed and unmanaged view. With the unmanaged view, there is no guarantee that referenced address is valid, only that it is a non-null pointer.


## Assignment Rules

Assignment rules cover the assignment operator as well as copy constructors. We aim at making all logically legal assignments possible, 
while intercepting illegal assignments if possible at compile time, otherwise at runtime.
In the following we use `DstType` and `SrcType` as the type of the destination view and source view respectively. 
`dst_view` and `src_view` refer to the runtime instances of the destination and source views, i.e.:
```c++
ScrType src_view(...);
DstType dst_view(src_view);
dst_view = src_view;
```

The following conditions must be met at and are evaluated at compile time:
 * `DstType::rank == SrcType::rank`
 * `DstType::non_const_value_type` is the same as `SrcType::non_const_value_type`
 * If `std::is_const<SrcType::value_type>::value == true` than `std::is_const<DstType::value_type>::value == true`.
 * `MemorySpaceAccess<DstType::memory_space,SrcType::memory_space>::assignable == true` 
 * If `DstType::dynamic_rank != DstType::rank` and `SrcType::dynamic_rank != SrcType::rank` than for each dimension `k` which is compile time for both it must be true that `dst_view.extent(k) == src_view.extent(k)`

Additionally the following conditions must be met at runtime: 
 * If `DstType::dynamic_rank != DstType::rank` than for each compile time dimension `k` it must be true that `dst_view.extent(k) == src_view.extent(k)`.
 
Furthermore there are rules which must be met if `DstType::array_layout` is not the same as `SrcType::array_layout`.
These rules only cover cases where both layouts are one of `LayoutLeft`, `LayoutRight` or `LayoutStride`
 * If neither `DstType::array_layout` nor `SrcType::array_layout` is `LayoutStride`: 
   * If `DstType::rank > 1` than `DstType::array_layout` must be the same as `SrcType::array_layout`.
 * If either `DstType::array_layout` or `SrcType::array_layout` is `LayoutStride`
   * For each dimension `k` it must hold that `dst_view.extent(k) == src_view.extent(k)`

### Assignment Examples
 ```c++
  View<int*>       a1 = View<int*>("A1",N);     // OK
  View<int**>      a2 = View<int*[10]>("A2",N); // OK
  View<int*[10]>   a3 = View<int**>("A3",N,M);  // OK if M == 10 otherwise runtime failure
  View<const int*> a4 = a1;                     // OK
  View<int*>       a5 = a4;                     // Error: const to non-const assignment
  View<int**>      a6 = a1;                     // Error: Ranks do not match
  View<int*[8]>    a7 = a3;                     // Error: compile time dimensions do not match
  View<int[4][10]> a8 = a3;                     // OK if N == 4 otherwise runtime failure
  View<int*, LayoutLeft>    a9  = a1;           // OK since a1 is either LayoutLeft or LayoutRight
  View<int**, LayoutStride> a10 = a8;           // OK
  View<int**>               a11 = a10;          // OK
  View<int*, HostSpace> a12 = View<int*, CudaSpace>("A12",N); // Error: non-assignable memory spaces
  View<int*, HostSpace> a13 = View<int*, CudaHostPinnedSpace>("A13",N); // OK
 ```

## Examples


```c++
#include<Kokkos_Core.hpp>
#include<cstdio> 

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc,argv);

   int N0 = atoi(argv[1]);
   int N1 = atoi(argv[2]);

   Kokkos::View<double*> a("A",N0);
   Kokkos::View<double*> b("B",N1);

   Kokkos::parallel_for("InitA", N0, KOKKOS_LAMBDA (const int& i) {
     a(i) = i;
   });

   Kokkos::parallel_for("InitB", N1, KOKKOS_LAMBDA (const int& i) {
     b(i) = i;
   });

   Kokkos::View<double**,Kokkos::LayoutLeft> c("C",N0,N1);
   {
     Kokkos::View<const double*> const_a(a);
     Kokkos::View<const double*> const_b(b);
     Kokkos::parallel_for("SetC", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{N0,N1}),
       KOKKOS_LAMBDA (const int& i0, const int& i1) {
       c(i0,i1) = a(i0) * b(i1);
     });
   }

   Kokkos::finalize();
}
```
