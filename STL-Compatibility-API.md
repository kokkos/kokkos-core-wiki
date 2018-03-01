## STL Compatibility Issues

Kokkos developers strive to implement the Kokkos macros in a manner compatible with the latest versions of the C++ language and with the C++ Standard Library (STL). However deviations from this approach occur when necessary, in particular, for the following the STL versions (std::*) do not work on GPUs. This section documents several of those deviations and provides usage guidance for developers in the following specific areas; each can be accessed by selecting the appropriate link.

----------

|Type  |Description                  |
|:-----|:----------------------------|
|[Kokkos::Complex](Kokkos%3A%3AComplex) | Kokkos::Complex Usage |
|[Kokkos::Array](Kokkos%3A%3AArray) | Kokkos::Array Usage |
|[Kokkos::pair](Kokkos%3A%3Apair)   | Kokkos::pair Usage |
