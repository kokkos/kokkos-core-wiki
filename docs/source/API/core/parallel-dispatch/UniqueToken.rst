``UniqueToken``
===============

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``Kokkos_Core.hpp``


Description
------------

``UniqueToken`` is a portable way to acquire a unique ID for calling a thread (``thread-id`` is not portable execution environments).  ``UniqueToken`` is thus analogous to ``thread-id``, and has a ``UniqueTokenScope`` template parameter (default: ``Instance``, but can be ``Global``).    

Interface
---------

.. cppkokkos:class:: template <class ExecutionSpace, UniqueTokenScope :: Global> UniqueToken



Parameters
-----------

*  ``ExecutionSpace``:  See `Execution Spaces <../execution_spaces.html>`_

.. note::
   In a parallel region, before the main computation, a pool of ``UniqueToken`` (integer) Id is generated, and each Id is released following iteration.

.. warning::
   ``UniqueToken <ExecutionSpace> token`` *can* be called inside a parallel region, *but* must be released at the end of *each* iteration.


*  ``UniqueTokenScope``:  defaults to ``Instance``, but ``Global`` can be employed when thread awareness is needed for more than one ``ExecutionSpace`` instance, as in the case of submitting concurrent kernels to CUDA streams.


Constructors
-------------
  .. cppkokkos:function:: UniqueToken(size_t max_size, ExecutionSpace execution_space, UniqueTokenScope :: Global)




Examples
---------

.. code-block:: cpp

  // UniqueToken on an Execution Space Instance
  UniqueToken < ExecutionSpace > token ;
  int number_of_uniqe_ids = token.size ();
  RandomGenPool pool ( number_of_unique_ids , seed );

  parallel_for ("L", N, KOKKOS_LAMBDA ( int i) {
    int id = token . acquire ();
    RandomGen gen = pool (id );
    // Computation Body
    token . release (id );
    });

  // Submitting concurrent kernels to (e.g., CUDA) streams

  void foo () {
  UniqueToken < ExecSpace , UniqueTokenScope :: Global > token_foo ;
  parallel_for ("L", RangePolicy < ExecSpace >( stream1 ,0,N), functor_a ( token_foo ));}

  void bar () {
  UniqueToken < ExecSpace , UniqueTokenScope :: Global > token_bar ;
  parallel_for ("L", RangePolicy < ExecSpace >( stream2 ,0,N), functor_b ( token_bar ));}


