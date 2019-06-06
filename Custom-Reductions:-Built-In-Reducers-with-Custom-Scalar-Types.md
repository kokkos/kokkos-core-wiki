In order to use a Custom Scalar Type with Built-in reductions, the following requirements must be fulfilled.

   * An initialization function must be provided via a specialization of the Kokkos::reduction_identity<T> class.  
   * Operators required for applied reduction class must be implemented.
   * The class / struct must either use the default copy constructor or have a specific copy constructor 
     implemented. 