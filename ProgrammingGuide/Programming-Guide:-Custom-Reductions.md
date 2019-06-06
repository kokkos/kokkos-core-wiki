# Chapter 9

# Custom Reductions

As described previously Kokkos reductions perform a "sum" reduction by default. But while that is the most common type of reduction it is not the only one required by more complex applications. Kokkos provides the "Reducer" concept to accommodate custom reductions. 

A Reducer is a class which provides the necessary information to join (reduce) two values, knows how to initialize thread private variables and where to store the final result of a reduction. Depending on your situation, you may need to write more or less code specialization for a custom reduction. 

## **[[Built-In Reducers|Custom Reductions: Build-In Reducers]]**
If you simply want to do a common operation, such as finding the minimum for a build-in scalar type, no custom code is required. 

## **[[Built-In Reducers with Custom Scalar Types|Custom-Reductions:-Built-In-Reducers-with-Custom-Scalar-Types]]**
In case you want to use your own scalar types, say a custom struct, that scalar type must provide the necessary operators (for example comparison operators for min-reductions). You must also provide a specialization of the reduction_identity class. 

## **[[Custom Reducers|Custom Reductions: Custom Reducers]]**
For completely arbitrary reductions, you must provide an implementation of a Reducer class. 