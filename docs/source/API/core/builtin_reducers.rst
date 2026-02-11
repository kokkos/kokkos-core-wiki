Built-in Reducers
=================

`ReducerConcept <builtinreducers/ReducerConcept.html>`__ provides the concept for Reducers.

Reducer objects used in conjunction with `parallel_reduce <parallel-dispatch/parallel_reduce.html>`__

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Reducer
     - Description
   * - `BAnd <builtinreducers/BAnd.html>`__
     - Binary 'And' reduction
   * - `BOr <builtinreducers/BOr.html>`__
     - Binary 'Or' reduction
   * - `LAnd <builtinreducers/LAnd.html>`__
     - Logical 'And' reduction
   * - `LOr <builtinreducers/LOr.html>`__
     - Logical 'Or' reduction
   * - `Max <builtinreducers/Max.html>`__
     - Maximum reduction
   * - `MaxFirstLoc <builtinreducers/MaxFirstLoc.html>`__
     - Reduction providing maximum and its smallest associated index
   * - `MaxLoc <builtinreducers/MaxLoc.html>`__
     - Reduction providing maximum and an associated index
   * - `Min <builtinreducers/Min.html>`__
     - Minimum reduction
   * - `MinFirstLoc <builtinreducers/MinFirstLoc.html>`__
     - Reduction providing minimum and its smallest associated index
   * - `MinLoc <builtinreducers/MinLoc.html>`__
     - Reduction providing minimum and an associated index
   * - `MinMax <builtinreducers/MinMax.html>`__
     - Reduction providing both minimum and maximum
   * - `MinMaxFirstLastLoc <builtinreducers/MinMaxFirstLastLoc.html>`__
     - Reduction providing both minimum at its smallest index & maximum at its largest index
   * - `MinMaxLoc <builtinreducers/MinMaxLoc.html>`__
     - Reduction providing both minimum and maximum and associated indices
   * - `Prod <builtinreducers/Prod.html>`__
     - Multiplicative reduction
   * - `Sum <builtinreducers/Sum.html>`__
     - Sum reduction


:cpp:struct:`reduction_identity` defines the neutral elements (identity values)
for various reduction operations. Specializing it is crucial for enabling
built-in reducers to work with user-defined types.

`Reduction Scalar Types <builtinreducers/ReductionScalarTypes.html>`__ are template classes for storage for reducers.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./builtinreducers/ReducerConcept
   ./builtinreducers/BAnd
   ./builtinreducers/BOr
   ./builtinreducers/LAnd
   ./builtinreducers/LOr
   ./builtinreducers/Max
   ./builtinreducers/MaxFirstLoc
   ./builtinreducers/MaxLoc
   ./builtinreducers/Min
   ./builtinreducers/MinFirstLoc
   ./builtinreducers/MinLoc
   ./builtinreducers/MinMax
   ./builtinreducers/MinMaxFirstLastLoc
   ./builtinreducers/MinMaxLoc
   ./builtinreducers/Prod
   ./builtinreducers/Sum
   ./builtinreducers/ReductionScalarTypes
   ./builtinreducers/reduction_identity
