## STL Compatibility Issues

Kokkos developers strive to implement the Kokkos macros in a manner compatible with the latest versions of the C++ language and with the C++ Standard Library (STL). However deviations from this approach occur when necessary, as is the case for the following STL classes (std::*). The STL class does not work properly on GPUs so parallel Kokkos classes have been developed. This section documents the specific deviations and provides usage guidance for developers. Select the links below to see the details.

----------

|Type  |Description                  |
|:-----|:----------------------------|
|[Kokkos::Array](Kokkos%3A%3AArray) | Kokkos::Array Usage |
|[Kokkos::Complex](Kokkos%3A%3AComplex) | Kokkos::Complex Usage |
|[Kokkos::pair](Kokkos%3A%3Apair)   | Kokkos::pair Usage |
