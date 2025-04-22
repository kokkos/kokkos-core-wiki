Moving code from requiring ``Kokkos_ENABLE_CUDA_UVM`` to using ``SharedSpace``
==============================================================================

.. role:: cpp(code)
   :language: cpp

.. _SharedSpace: ../API/core/memory_spaces.html#kokkos-sharedspace
.. |SharedSpace| replace:: ``SharedSpace``

.. _ExecutionSpace: ../API/core/execution_spaces.html#kokkos-executionspaceconcept
.. |ExecutionSpace| replace:: ``ExecutionSpace``

.. _SharedHostPinnedSpace: ../API/core/memory_spaces.html#kokkos-sharedhostpinnedspace
.. |SharedHostPinnedSpace| replace:: ``SharedHostPinnedSpace``

.. _KokkosSharedSpace: ../API/core/memory_spaces.html#kokkos-sharedspace
.. |KokkosSharedSpace| replace:: ``Kokkos::SharedSpace``

With Kokkos 4.0 ``Kokkos_ENABLE_CUDA_UVM`` is deprecated and can only be used with ``Kokkos_ENABLE_DEPRECATED_CODE_4``. The main reason for the deprecation was, that using the option changed the ``memory_space`` of the ``Cuda`` ``ExecutionSpace``. This lead to several problems. For example: The driver is allowed to move chunks of this memory to the device or host depending on the access at any time without notice.
The accesses in ``parallel_for``, ``parallel_reduce``, or ``parallel_scan`` do not occur in any guaranteed order and furthermore depend on other kernels running on the same GPU. This makes debugging tedious. Especially, if the memory an allocation resides in is not apparent but dependent on the options when running ``cmake``.

The alternative
---------------

We introduced a new alias named |SharedSpace|_ in Kokkos 4.0. This always points to memory that is accessible by every |ExecutionSpace|_ and is migrated without user interaction to the accessing ``ExecutionSpace`` on demand. After migration the memory is accessed locally.
Using the alias e.g. in ``Views`` is expressive and thus easier to read. Furthermore, it is portable to every backend that can automatically migrate memory between ``ExecutionSpaces``.
Furthermore, we introduced the alias |SharedHostPinnedSpace|_ which points to memory that is accessible by all enabled ``ExecutionSpaces`` but always resides in the memory of the host.

The transition
--------------

Basically it comes down to spelling |KokkosSharedSpace|_ as a template argument in all allocations.
Below is an example of a transition:

* Code requiring ``Kokkos_ENABLE_CUDA_UVM`` at configure time (until 4.0)

.. code-block:: cpp

   #include <Kokkos_Core.hpp>

   int main (){
     Kokkos::initialize();
     {
       unsigned int N = 100;
       Kokkos::View<int*> myView("myView",N);
       void* c_style_memory = Kokkos::kokkos_malloc("c_style_alloc",N*sizeof(double));

       ...

       Kokkos::kokkos_free(c_style_memory);
     }
     Kokkos::finalize();
     return 0;
   }

* Code using ``SharedSpace`` (since 4.0)

.. code-block:: cpp

   #include <Kokkos_Core.hpp>

   int main (){
     Kokkos::initialize();
     {
       static_assert(Kokkos::has_shared_space(),"code only works on backends with SharedSpace");

       unsigned int N = 100;
       Kokkos::View<int*,Kokkos::SharedSpace> myView("myView",N);
       void* c_style_memory = Kokkos::kokkos_malloc<Kokkos::SharedSpace>("c_style_alloc",N*sizeof(double));

       ...

       Kokkos::kokkos_free<Kokkos::SharedSpace>(c_style_memory);
     }
     Kokkos::finalize();
     return 0;
   }
