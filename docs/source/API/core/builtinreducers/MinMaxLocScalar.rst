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

Interface
---------

.. cpp:struct::  template<class Scalar, class Index> MinMaxLocScalar

   :tparam Scalar: The data type of the value being reduced.
   :tparam Index: The data type of the locations (indices) of the values.

   .. rubric:: Data members

   .. cpp:var:: Scalar min_val

      The reduced minimum value.

   .. cpp:var:: Scalar max_val

      The reduced maximum value.

   .. cpp:var:: Index min_loc

      The location (iteration index) of the minimum value

   .. cpp:var:: Index max_loc

      The location (iteration index) of the maximum value
