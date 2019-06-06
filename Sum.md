 * ```c++
   template<class Scalar, class Space>
   class Sum;
   ```
   Uses the addition operation to combine partial results;
   * `Sum<T,S>::value_type` is `T`
   * `Sum<T,S>::result_view_type` is `Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>`
   * Requires: `Scalar` has `operator =` and `operator +=` defined. `Kokkos::reduction_identity<Scalar>::sum()` is a valid expression. 