Built-in Reducers
=================

:doc:`ReducerConcept <builtinreducers/ReducerConcept>` provides the concept for Reducers.

Reducer objects used in conjunction with :doc:`parallel_reduce <parallel-dispatch/parallel_reduce>`

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Reducer
     - Description
   * - :doc:`BAnd <builtinreducers/BAnd>`
     - Binary 'And' reduction
   * - :doc:`BOr <builtinreducers/BOr>`
     - Binary 'Or' reduction
   * - :doc:`LAnd <builtinreducers/LAnd>`
     - Logical 'And' reduction
   * - :doc:`LOr <builtinreducers/LOr>`
     - Logical 'Or' reduction
   * - :doc:`Max <builtinreducers/Max>`
     - Maximum reduction
   * - :doc:`MaxLoc <builtinreducers/MaxLoc>`
     - Reduction providing maximum and an associated index
   * - :doc:`Min <builtinreducers/Min>`
     - Minimum reduction
   * - :doc:`MinLoc <builtinreducers/MinLoc>`
     - Reduction providing minimum and an associated index
   * - :doc:`MinMax <builtinreducers/MinMax>`
     - Reduction providing both minimum and maximum
   * - :doc:`MinMaxLoc <builtinreducers/MinMaxLoc>`
     - Reduction providing both minimum and maximum and associated indices
   * - :doc:`Prod <builtinreducers/Prod>`
     - Multiplicative reduction
   * - :doc:`Sum <builtinreducers/Sum>`
     - Sum reduction


:cpp:struct:`reduction_identity` defines the neutral elements (identity values)
for various reduction operations. Specializing it is crucial for enabling
built-in reducers to work with user-defined types.

:doc:`Reduction Scalar Types <builtinreducers/ReductionScalarTypes>` are template classes for storage for reducers.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./builtinreducers/ReducerConcept
   ./builtinreducers/BAnd
   ./builtinreducers/BOr
   ./builtinreducers/LAnd
   ./builtinreducers/LOr
   ./builtinreducers/Max
   ./builtinreducers/MaxLoc
   ./builtinreducers/Min
   ./builtinreducers/MinLoc
   ./builtinreducers/MinMax
   ./builtinreducers/MinMaxLoc
   ./builtinreducers/Prod
   ./builtinreducers/Sum
   ./builtinreducers/ReductionScalarTypes
   ./builtinreducers/reduction_identity
