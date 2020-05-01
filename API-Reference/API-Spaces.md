Kokkos contains two `Space` Concepts: [`MemorySpace`](Kokkos%3A%3AMemorySpaceConcept) and [`ExecutionSpace`](Kokkos%3A%3AExecutionSpaceConcept).
Concrete instances of these two concepts are used to allocate data and dispatch work. Their relationship is described through 
the [`Kokkos::SpaceAccessibility`](Kokkos%3A%3ASpaceAccessibility) trait.

## Execution Spaces

  The Concept is documented in [`ExecutionSpace`](Kokkos%3A%3AExecutionSpaceConcept).

  * [`Cuda`](Kokkos%3A%3ACuda)
  * [`HPX`](Kokkos%3A%3AHPX)
  * [`OpenMP`](Kokkos%3A%3AOpenMP)
  * [`Serial`](Kokkos%3A%3ASerial)
  * [`Threads`](Kokkos%3A%3AThreads)

## Memory Spaces

  The Concept is documented in [`MemorySpace`](Kokkos%3A%3AMemorySpaceConcept).

  * [`CudaSpace`](Kokkos%3A%3ACudaSpace)
  * [`CudaHostPinnedSpace`](Kokkos%3A%3ACudaHostPinnedSpace)
  * [`CudaUVMSpace`](Kokkos%3A%3ACudaUVMSpace)
  * [`HostSpace`](Kokkos%3A%3AHostSpace)

## Facilities

  * [`is_execution_space`](Kokkos%3A%3AExecutionSpaceConcept)
  * [`is_memory_space`](Kokkos%3A%3AMemorySpaceConcept)
  * [`SpaceAccessibility`](Kokkos%3A%3ASpaceAccessibility)

## C-style memory management
  * [`kokkos_malloc`](Kokkos%3A%3Akokkos_malloc)
  * [`kokkos_realloc`](Kokkos%3A%3Akokkos_realloc)
  * [`kokkos_free`](Kokkos%3A%3Akokkos_free)
