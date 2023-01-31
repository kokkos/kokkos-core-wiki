``MinMaxScalar``
================

.. role::cpp(code)
    :language: cpp

Template class for storing the min and max values for min/max reducers. Should be accessed via ``::value_type`` defined for particular reducer.

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

Public Members
--------------

Variables
~~~~~~~~~

* ``min_val``: Scalar minimum Value.
* ``max_val``: Scalar maximum Value.

Assignment operators
~~~~~~~~~~~~~~~~~~~~

* ``void operator = (const MinMaxScalar& rhs);`` assign ``min_val`` and ``max_val`` from ``rhs``;
