# `LayoutStride`

Header File: `Kokkos_Layout.hpp`

This Kokkos Layout, when provided to a multidimensional View, lays out memory with an arbitrary stride.

Most frequently encountered when taking a noncontiguous subview of some larger view.

Usage: 

```c++
Kokkos::View<float***> full_mesh; // an entire mesh
Kokkos::View<float**, Kokkos::LayoutStride> mesh_subcomponent;
mesh_subcomponent = Kokkos::subview(full_mesh,Kokkos::ALL(), 0, Kokkos::ALL()); // take x and z components
```

## Synopsis 
```c++
struct LayoutStride {
    typedef LayoutStride array_layout;
  
    size_t dimension[ARRAY_LAYOUT_MAX_RANK];
    size_t stride[ARRAY_LAYOUT_MAX_RANK];
  
    enum { is_extent_constructible = false };
  
    LayoutStride(LayoutStride const&) = default;
    LayoutStride(LayoutStride&&)      = default;
    LayoutStride& operator=(LayoutStride const&) = default;
    LayoutStride& operator=(LayoutStride&&) = default;
  
    template <typename iTypeOrder, typename iTypeDimen>
    KOKKOS_INLINE_FUNCTION static LayoutStride order_dimensions(
        int const rank, iTypeOrder const* const order,
        iTypeDimen const* const dimen);

    KOKKOS_INLINE_FUNCTION
    explicit constexpr LayoutStride(size_t N0 = 0, size_t S0 = 0, size_t N1 = 0,
                                    size_t S1 = 0, size_t N2 = 0, size_t S2 = 0,
                                    size_t N3 = 0, size_t S3 = 0, size_t N4 = 0,
                                    size_t S4 = 0, size_t N5 = 0, size_t S5 = 0,
                                    size_t N6 = 0, size_t S6 = 0, size_t N7 = 0,
                                    size_t S7 = 0);
};
```

## Public Class Members

  * `dimension`

    An array containing the size of each dimension of the Layout

  * `stride`

    An array containing the stride for each dimension of the Layout
   
## Typedefs
   
 * `array_layout`

    A tag signifying that this models the Layout concept

## Enums

  * `is_extent_constructible`

    A boolean enum to allow detection that this class is extent constructible

### Constructors

  * `LayoutStride(LayoutStride const&) = default;`

    Default copy constructor, element-wise copies the other Layout

  * `LayoutStride(LayoutStride&&) = default;`
 
    Default move constructor, element-wise moves the other Layout

  * 
```c++
KOKKOS_INLINE_FUNCTION
explicit constexpr LayoutStride(size_t N0 = 0, size_t S0 = 0, size_t N1 = 0,
                              size_t S1 = 0, size_t N2 = 0, size_t S2 = 0,
                              size_t N3 = 0, size_t S3 = 0, size_t N4 = 0,
                              size_t S4 = 0, size_t N5 = 0, size_t S5 = 0,
                              size_t N6 = 0, size_t S6 = 0, size_t N7 = 0,
                              size_t S7 = 0);
```
  
  Constructor that takes in up to 8 sizes, to set the sizes of the corresponding dimensions of the Layout

### Assignment operators

  * LayoutStride& operator=(LayoutStride const&) = default;

    Default copy assignment, element-wise copies the other Layout

  * LayoutStride& operator=(LayoutStride&&) = default;

    Default move assignment, element-wise moves the other Layout

### Functions

  * ```c++
        KOKKOS_INLINE_FUNCTION static LayoutStride order_dimensions(
        int const rank, iTypeOrder const* const order,
        iTypeDimen const* const dimen);
    ``` 
    
    Calculates the strides given ordered dimensions

## Example

Creating a 3D unmanaged strided view around a ptr. (You can also just have a view allocate itself by providing a label)

```c++
#include<Kokkos_Core.hpp>
int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
     // Some storage
     int* ptr = new int[80];
     // Creating a layout object
     Kokkos::LayoutStride layout(3,1,3,5,4,20);
     // Create a unmanaged view from a pointer and a layout
     Kokkos::View<int***, Kokkos::LayoutStride, Kokkos::HostSpace> a(ptr,layout);
     
     // Get strides
     int strides[8];
     a.stride(strides);
     
     // Print extents and strides
     printf("Extents: %d %d %d\n",a.extent(0),a.extent(1),a.extent(2));
     printf("Strides: %i %i %i\n",strides[0],strides[1],strides[2]);
     
     // delete storage
     delete [] ptr;
  }
  Kokkos::finalize();
}
```

Output:
```
Extents: 3 3 4
Strides: 1 5 20
```
