_Kokkos::Complex_ math functions are defined for the same builtin data types specialized for C++ _std::complex_ functions:

```float, double, long double```

Developers will have greater success in the long term by following the general strategy outlined below.

```c++
   using std::sqrt;
   Scalar x = ...
   Scalar y = sqrt(x);
```

The using _std::sqrt_ will handle the case when Scalar is any builtin type. When Scalar is any derived type, such as _Kokkos::complex_, the right overload will be found through argument-dependent lookup, as long as the overload is placed in the same namespace that Scalar is in (which it is for _Kokkos::complex_).
