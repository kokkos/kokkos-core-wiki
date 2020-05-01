Data management is a critical part of any program. The main facility in Kokkos is the `Kokkos::View`. 
The following facilities are available:

  * [`View`](Kokkos%3A%3AView): The main Kokkos data structure, a multi dimensional memory space and layout aware array.
  * [`subview`](Kokkos%3A%3Asubview): Getting slices from a `Kokkos::View`.
  * [`realloc`](Kokkos%3A%3Arealloc): Reallocating a `Kokkos::View`.
  * [`resize`](Kokkos%3A%3Aresize): Resizinc a `Kokkos::View`.
  * [`create_mirror`](Kokkos%3A%3Acreate_mirror): Creating a copy of a `Kokkos::View` in a different memory space.
  * [`create_mirror_view`](Kokkos%3A%3Acreate_mirror): Creating a copy of a `Kokkos::View` in a different memory space, if the original view is not accessible in that space.
  * [`LayoutLeft`](Kokkos%3A%3ALayoutLeft): Memory Layout matching Fortran 
  * [`LayoutRight`](Kokkos%3A%3ALayoutRight): Memory Layout matching C 
  * [`LayoutStride`](Kokkos%3A%3ALayoutStride): Memory Layout for arbitrary strides 

