# Overlapping Host and Device work 

When using architectures that allow host execution to concur with device execution Kokkos supports 
overlapping host operations with device operations which can produce significant speedup depending 
on the algorithm.  This use case describes the conditions and the design of algorithms that take advange of
overlapping device execution with host execution.

## Actors
 - Algorithm with different set of kernels where some are best executed on the host and some 
are better executed on an accelerator device
 - Algorithm where communication or serialization operations can be staggered with computational kernels
 - Algorithm where work can be divied between host and device without contention to resources
 
## Subjects
 - Kokkos Execution Spaces
 - Kokkos Execution Policies
 - Kokkos Memory Spaces

## Assumptions
 - There is little or no contention with host accessible memory while a device kernel is executing
    
## Constraints
 - Kernels are non-blocking
    
## Preconditions
 - Execution "kernel" implemented in the form of C++ functor
    
## Usage Pattern 1 - overapping computational kernels

```

  |--- Allocate Device and Host Memory
  |--- Initialize Host and Device Memory
  |------------------------------------------------
  |- Perform host operation 0
  |--- iteration loop -----------------------------
    |->----------- global barrier -----------------
    |  |- Synchronize host and device data
    |  |- Perform device operation N \
    |  |- Perform host operation N+1 / asynchronous
    |-<--------------------------------------------
       
```

## Example Code
Perform setup of host data needed for iteration n+1 while device is performing
operation on iteration n

```c++
  typedef double value_type;
  typedef Kokkos::OpenMP   HostExecSpace;
  typedef Kokkos::Cuda     DeviceExecSpace;
  typedef Kokkos::RangePolicy<DeviceExecSpace>  device_range_policy;
  typedef Kokkos::RangePolicy<HostExecSpace>    host_range_policy;
  typedef Kokkos::View<double*, Kokkos::CudaSpace>   ViewVectorType;
  typedef Kokkos::View<double**, Kokkos::CudaSpace>  ViewMatrixType;

  // Setup data on host  
  // use parameter xVal to demonstrate variability between iterations
  void init_src_views(ViewVectorType::HostMirror p_x,
                      ViewMatrixType::HostMirror p_A,
                      const value_type xVal ) {
    
    Kokkos::parallel_for( "init_A", host_range_policy(0,N), [=] ( int i ) {
      for ( int j = 0; j < M; ++j ) {
        h_A( i, j ) = 1;
      }
    });

    Kokkos::parallel_for( "init_x", host_range_policy(0,M), [=] ( int i ) {
      h_x( i ) = xVal;
    });
  }
  
  ViewVectorType y( "y", N );
  ViewVectorType x( "x", M );
  ViewMatrixType A( "A", N, M );
  
  ViewVectorType::HostMirror h_y = Kokkos::create_mirror_view( y );
  ViewVectorType::HostMirror h_x = Kokkos::create_mirror_view( x );
  ViewMatrixType::HostMirror h_A = Kokkos::create_mirror_view( A );
  
  for ( int repeat = 0; repeat < nrepeat; repeat++ ) {
    init_src_views( h_x, h_A, repeat+1);  // setup data for next device launch
  
    Kokkos::fence(); // barrier used to synch between device and host before copying data
    
    // Deep copy host data needed for this iteration to device.
    Kokkos::deep_copy( h_y, h );
    Kokkos::deep_copy( x, h_x );
    Kokkos::deep_copy( A, h_A );  // implicit barrier

    // Application: y=Ax
    Kokkos::parallel_for( "yAx", device_range_policy( 0, N ), 
                                KOKKOS_LAMBDA ( int j ) {
      double temp2 = 0;
      for ( int i = 0; i < M; ++i ) {
        temp2 += A( j, i ) * x( i );
      }

      y( j ) = temp2;
    } );
    
    // note that there is no barrier here, so the host thread will loop
    // back around and call ini_src_views while the kernel is still running
   }

  
```

<b>Important note</b>:  In theory, the order in which the host kernel and the device kernel are
launched is not important, but in practice the device kernel must be launched first.  Most 
host backends do not leave a "main" thread free while the kernel is running.  Once the 
host parallel kernel is launched, the main thread is occupied until that thread's 
contribution to the kernel is complete.  Because the device execution is in a different context, 
the host thread is free immediately after the kernel is launched.  Attention must 
also be paid to the contract associated with the parallel execution pattern.  If the pattern
requires a synchronization prior to completion (such as a reduction), then there is no opportunity to 
overlap host and device operations.  Thus, taking advantage of a host/device
overlapping pattern may require modifications to the overall algorithm.


## Usage pattern 2 - perform serialized operation on host while device is executing kernel

```
  |--- Allocate Device and Host Memory
  |--- Initialize Host and Device Memory
  |---------- global barrier ------------------------------
  |- Synchronize host and device data
  |--------------------------------------------------------
    |->|- Perform device operation N         \ asynchronous
    |  |- Serialize host data from N to disk /
    |  |------------ global barrier -----------------------
    |  |- Synchronize host and device data for start of N+1
    |-<----------------------------------------------------

```

Data serialized to disk is behind by 1 iteration, but it can be performed 
asynchronously with the device operation.  Device data for N+1 is copied after
iteration N which is why the barrier is need before the synchronization

## example code

```c++
  typedef Kokkos::RangePolicy<>    range_policy;
  typedef Kokkos::View<double*>    ViewVectorType;

  ViewVectorType V_r;
  ViewVectorType V_r1;
  ViewVectorType::HostMirror h_V = Kokkos::create_mirror_view( y );

  get_initial_state(h_V); // function to initialize V on host
  
  Kokkos::deep_copy(V_r, h_V);
  Kokkos::deep_copy(V_r1, h_V)

  for (int r = 0; r < R; r++) {
  
     Kokkos::parallel_for(range_policy(0,N), KOKKOS_LAMBDA (int i) {
        V_r1(i) = get_RHS_func(V_r);  //return V_r1(i) for RHS from V_r
     });
  
     serialize_state(h_V); // serialize data still in host view_r
     
     Kokkos::fence();  // synchronize between host and device
     
     Kokkos::deep_copy(h_V, V_r1);  // update for next iteration
     Kokkos::deep_copy(V_r, h_V);
     
  }
```

