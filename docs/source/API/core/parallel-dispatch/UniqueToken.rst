``UniqueToken``
===============

Header File: ``Kokkos_Core.hpp``


Description
------------

``UniqueToken`` is a portable way to acquire a unique ID for a thread (``thread-id`` is not portable across execution environments).  The unique ID is scoped by the ``UniqueTokenScope`` template parameter (defaults to ``Instance``, but can be set to ``Global``).

Interface
---------

.. code-block:: cpp 

    template <class ExecutionSpace = DefaultExecutionSpace, UniqueTokenScope = UniqueTokenScope::Instance> UniqueToken



Parameters
-----------

*  ``ExecutionSpace``:  See `Execution Spaces <../execution_spaces.html>`_

*  ``UniqueTokenScope``:  defaults to ``Instance``, and every instance is independent of another.  In contrast, ``Global`` uses one set of unique IDs for all instances.

.. note::
   In a parallel region, before the main computation, a pool of ``UniqueToken`` (integer) ID is generated.  A generated ID is released following iteration (see ``void release(size_t idx)`` below).


Constructors
-------------

  .. code-block:: cpp

     UniqueToken(size_t max_size, ExecutionSpace execution_space = ExecutionSpace{})
     // Scope is instance
  
  .. code-block:: cpp
     
     UniqueToken(ExecutionSpace execution_space = ExecutionSpace{}); 
     // Scope is instance or global



Public Member Functions
------------------------
     
 .. code-block:: cpp
    
    size_t size()
    // Returns the size of the token pool    
 
 .. code-block:: cpp

    size_t acquire()
    // Returns the token for the executing tread     

 .. code-block:: cpp

    void release(size_t idx)
    // Releases the passed token

.. warning::
   Acquired tokens *must* be released at the end of the parallel region in which they were acquired
 



Examples
---------

.. code-block:: cpp

  // UniqueToken on an Execution Space Instance
  UniqueToken < ExecutionSpace > token ;
  int number_of_uniqe_ids = token.size ();
  RandomGenPool pool ( number_of_unique_ids , seed );

  parallel_for ("L", N, KOKKOS_LAMBDA ( int i) {
    auto id = token.acquire ();
    RandomGen gen = pool (id);
    // Computation Body
    token.release (id);
    });

  // Submitting concurrent kernels to (e.g., CUDA) streams

  void foo () {
    UniqueToken < ExecSpace , UniqueTokenScope :: Global > token_foo ;
    parallel_for ("L", RangePolicy < ExecSpace >( stream1 ,0,N), functor_a ( token_foo ));}

  void bar () {
    UniqueToken < ExecSpace , UniqueTokenScope :: Global > token_bar ;
    parallel_for ("L", RangePolicy < ExecSpace >( stream2,0,N), functor_b ( token_bar ));}
