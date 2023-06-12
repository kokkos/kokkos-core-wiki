``MinMaxScalar``
================

.. role::cpp(code)
    :language: cpp

Template class for storing the min and max values for min/max reducers.
Should be accessed via ``::value_type`` defined for a particular reducer.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   MinMax<T,S>::value_type result;
   parallel_reduce(N,Functor,MinMax<T,S>(result));
   T minValue = result.min_val;
   T maxValue = result.max_val;

Synopsis
--------

.. code-block:: cpp

   template<class Scalar>
   struct MinMaxScalar{
     Scalar min_val;
     Scalar max_val;

     void operator = (const MinMaxScalar& rhs);
   };


Interface
---------

.. cppkokkos:struct:: template<class Scalar> MinMaxScalar

   .. rubric:: Public Members

   .. cppkokkos:member:: Scalar min_val

      Scalar minimum Value.

   .. cppkokkos:member:: Scalar max_val

      Scalar maximum Value.

   .. rubric:: Assignment Operator

   .. cppkokkos:function:: void operator = (const MinMaxScalar& rhs)

      Assign ``min_val`` and ``max_val`` from ``rhs``
