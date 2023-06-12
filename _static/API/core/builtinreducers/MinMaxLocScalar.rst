``MinMaxLocScalar``
===================

.. role::cpp(code)
    :language: cpp

Template class for storing the min and max values with indices for min/max location reducers.
Should be accessed via ``::value_type`` defined for a particular reducer.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   MinMaxLoc<T,I,S>::value_type result;
   parallel_reduce(N,Functor,MinMaxLoc<T,I,S>(result));
   T minValue = result.min_val;
   T maxValue = result.max_val;
   I minLoc = result.min_loc;
   I maxLoc = result.max_loc;

Synopsis
--------

.. code-block:: cpp

   template<class Scalar, class Index>
   struct MinMaxLocScalar{
     Scalar min_val;
     Scalar max_val;
     Index min_loc;
     Index max_loc;

     void operator = (const MinMaxLocScalar& rhs);
   };

Interface
---------

.. cppkokkos:struct:: template<class Scalar, class Index> MinMaxLocScalar

   .. rubric:: Public Members

   .. cppkokkos:member:: Scalar min_val

      Scalar minimum Value.

   .. cppkokkos:member:: Scalar max_val

      Scalar maximum Value.

   .. cppkokkos:member:: Index min_loc

      Minimum location(Index).

   .. cppkokkos:member:: Index max_loc

      Maximum location(Index).

   .. rubric:: Assignment Operator

   .. cppkokkos:function:: void operator = (const MinMaxLocScalar& rhs)

      Assign ``min_val``, ``max_val``, ``min_loc`` and ``max_loc`` from ``rhs``
