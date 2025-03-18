.. role:: cppkokkos(code)
    :language: cppkokkos

Multi-GPU Support
=================

Kokkos has support for launching kernels on multiple GPU devices within a single node from host on a single MPI rank for the Cuda, HIP, and SYCL backends.

Using this feature requires knowledge of backend specific API calls for setting devices, creating streams, and (potentially)
allocating device specific data.

Basic Usage
-----------

For allocating and launching kernels on a chosen device, the user is responsible for creating device specific streams to pass to backend constructor.
Once these are created, they can be passed as any other execution space instance to create policies, allocated managed views, asynchronous deep copies, etc. For creating unmanaged
views, the user must manually allocate device specific memory to pass to the view constructor.

Contructing Execution Space
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cuda
^^^^

.. code-block:: cpp

    // Query number of devices available
    int n_devices;
    cudaGetDeviceCount(&n_devices);

    // Choose some 0 <= N < n_devices
    int N = ...;

    // Create stream on device N
    cudaStream_t stream;
    cudaSetDevice(N);
    cudaStreamCreate(&stream);

    // Scope execution space to ensure stream
    // is destroyed *after* execution space
    {
      // Create execution space
      Kokkos::Cuda exec_space(stream);

      // Use execution space
      /* ... */
    }

    // Destroying stream after use
    cudaSetDevice(N);
    cudaStreamDestroy(stream);

HIP
^^^

.. warning:: Multi-GPU only supported for rocm 5.6 and later. Because of the lack of HIP API functions for querying a stream's device
             in pre rocm 5.6, constructing a ``Kokkos::HIP`` instance on the non-default device will not be able to throw an error,
             problems will just likely arise downstream.


.. code-block:: cpp

    // Query number of devices available
    int n_devices;
    hipGetDeviceCount(&n_devices);

    // Choose some 0 <= N < n_devices
    int N = ...;

    // Create stream on device N
    hipStream_t stream;
    hipSetDevice(N);
    hipStreamCreate(&stream);

    // Scope execution space to ensure stream
    // is destroyed *after* execution space
    {
      // Create execution space
      Kokkos::HIP exec_space(stream);

      // Use execution space
      /* ... */
    }

    // Destroying stream after use
    hipSetDevice(N);
    hipStreamDestroy(stream);

SYCL
^^^^

.. code-block:: cpp

    // Get list of devices available
    std::vector<sycl::device> gpu_devices =
      sycl::device::get_devices(sycl::info::device_type::gpu);

    // Choose some 0 <= N < gpu_devices.size()
    int N = ...;

    // Create a queue on device N.
    // Note: Kokkos requires SYCL queues to be "in_order"
    sycl::queue queue{gpu_devices[N], sycl::property::queue::in_order()};

    // Create execution space
    Kokkos::SYCL exec_space(queue);

    // Use execution space
    /* ... */

Notes
-----

- For ``parallel_reduce(..., const ReducerArgument&... reducer)``, it is important to pass rank-0 views for the ``reducer`` arguments
  instead of scalar types. Scalar reductions imply a fence on all execution spaces.
- When passing a stream to an execution space constructor, the user is expected the manage the lifetime of the stream. Only after the
  execution space is destroyed can the user safely destroy the stream.
