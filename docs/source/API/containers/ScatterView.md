# `ScatterView`

Header File: `Kokkos_ScatterView.hpp`

Usage: 

`ScatterView` can tranparently switch between **Atomic** and **Data Replication** - based scatter algorithms.  Recall, **Atomics** are thread-scalable, and depend on atomic performance.  In **Data Replication** implementations, every thread owns a copy of the output, are not thread-scalable, but are good for low (< 16) thread-count architectures.  `ScatterView` is a compile-time choice with backend-specific defaults and has a limited number of supported operations.  Typically, `ScatterView` wraps an existing [`View`](../core/view/view), allowing the **atomic** variant to work *without* additional allocation -- remember to contribute back to the original `View`!  If possible, reuse a `ScatterView`, as creating and destroying data duplicates are costly.

## Interface

```c++
KOKKOS_INLINE_FUNCTION int foo(int i) { return i; }
KOKKOS_INLINE_FUNCTION double bar(int i) { return i*i; }

Kokkos::View<double*> results("results", 1);
Kokkos::Experimental::ScatterView<double*> scatter(results);
Kokkos::parallel_for(1, KOKKOS_LAMBDA(int input_i) {
 auto access = scatter.access();
 auto result_i = foo(input_i);
 auto contribution = bar(input_i);
 access(result_i) += contribution;
});
Kokkos::Experimental::contribute(results, scatter);
```


## Synopsis 
```c++

template <typename DataType
        ,int Operation
        ,typename ExecSpace
        ,typename Layout
        ,int contribution
        >
class ScatterView<DataType
                  ,Layout
                  ,ExecSpace
                  ,Operation
                  ,{ScatterNonDuplicated,ScatterDuplicated}
                  ,contribution>
{
public:
 typedef Kokkos::View<DataType, Layout, ExecSpace> original_view_type;
 typedef typename original_view_type::value_type original_value_type;
 typedef typename original_view_type::reference_type original_reference_type;
 friend class ScatterAccess<DataType, Operation, ExecSpace, Layout, {ScatterNonDuplicated,ScatterDuplicated}, contribution, ScatterNonAtomic>;
 friend class ScatterAccess<DataType, Operation, ExecSpace, Layout, {ScatterNonDuplicated,ScatterDuplicated}, contribution, ScatterAtomic>;
 []:# (TODO: Deprecate types requiring `Kokkos::Impl..` )
 typedef typename Kokkos::Impl::Experimental::DuplicatedDataType<DataType, {Kokkos::LayoutRight,Kokkos::LayoutLeft}> data_type_info; // ScatterDuplicated only
 typedef typename data_type_info::value_type internal_data_type; // ScatterDuplicated only
 typedef Kokkos::View<internal_data_type, {Kokkos::LayoutRight,Kokkos::LayoutLeft}, ExecSpace> internal_view_type; // ScatterDuplicated only
 
 ScatterView();

 template <typename ReferenceType, typename ... Properties>
 ScatterView(View<ReferenceType, Properties...> const& );

 template <typename ... Dimensions>
 ScatterView(std::string const& name, Dimensions ... dims);

 []: # (TODO: Deprecate types requiring `Kokkos::Impl..` )
 template <typename... Properties, typename... Dimensions>
 ScatterView(::Kokkos::Impl::ViewCtorProp<Properties...> const& arg_prop, Dimensions... dims);

 template <int override_contrib = contribution>
 KOKKOS_FORCEINLINE_FUNCTION
 ScatterAccess<DataType, Operation, ExecSpace, Layout, ScatterNonDuplicated, contribution, override_contrib>
 access() const;

 original_view_type subview() const;

 template <typename DataType, typename ... Properties>
 void contribute_into(View<DataType, Properties...> const& dest) const;

 void reset();
 
 template <typename DataType, typename ... Properties>
 void reset_except(View<DataType, Properties...> const& view);

 void resize(const size_t n0 = 0,
          const size_t n1 = 0,
          const size_t n2 = 0,
          const size_t n3 = 0,
          const size_t n4 = 0,
          const size_t n5 = 0,
          const size_t n6 = 0,
          const size_t n7 = 0);

 void realloc(const size_t n0 = 0,
          const size_t n1 = 0,
          const size_t n2 = 0,
          const size_t n3 = 0,
          const size_t n4 = 0,
          const size_t n5 = 0,
          const size_t n6 = 0,
          const size_t n7 = 0);

protected:
 template <typename ... Args>
 KOKKOS_FORCEINLINE_FUNCTION
 original_reference_type at(Args ... args) const;
 
private:
 typedef original_view_type internal_view_type;
 internal_view_type internal_view;
};
```

## Public Class Members

### Typedefs
* `original_view_type`: Type of `View` passed to ScatterView constructor.
* `original_value_type`: Value type of the `original_view_type`.
* `original_reference_type`: Reference type of the `original_view_type`.
[]: # (ScatterDuplicated only)
* `data_type_info`: DuplicatedDataType, a newly created DataType that has a new runtime dimension which becomes the largest-stride dimension, from the given `View` DataType.
* `internal_data_type`: Value type of `data_type_info`.
* `internal_view_type`: A `View` type created from the `internal_data_type`.

### Constructors

 * ```c++
    ScatterView();
   ```
   Default constructor. Default constructs members.

 * ```c++
    ScatterView(View<ReferenceType, Properties...> const& );
   ```
   Constructor from a `Kokkos::View`. `internal_view` member is copy constructed from this input view.

 * ```c++
    ScatterView(std::string const& name, Dimensions ... dims);
   ```
   Constructor from variadic pack of dimension arguments. Constructs `internal_view` member.

 []: # (TODO: Deprecate types requiring `Kokkos::Impl..` )
 * ```c++
    ScatterView(::Kokkos::Impl::ViewCtorProp<Properties...> const& arg_prop, Dimensions ... dims);
   ```
   Constructor from variadic pack of properties and dimension arguments. Constructs `internal_view` member.
   This constructor allows specifying an execution space instance to be used by passing, e.g., 
   `Kokkos::view_alloc(exec_space, "label")` as first argument.

### Functions
  * ```c++
    constexpr bool is_allocated() const;
    ```
    Returns true if the `internal_view` points to a valid memory location.  This function works for both managed and unmanaged views. With the unmanaged view, there is no guarantee that referenced address is valid, only that it is a non-null pointer.

 * ```c++
    access() const;
   ```
   Use within a kernel to return a `ScatterAccess` member; this member accumulates a given thread's contribution to the reduction.

 * ```c++
    subview() const;
   ```
   Return a subview of a `ScatterView`.

 * ```c++
    contribute_into(View<DataType, Properties...> const& dest) const;
   ```
   Contribute `ScatterView` array's results into the input View `dest`.

 * ```c++
    reset();
   ```
   Performs reset on destination array.

 * ```c++
    reset_except(View<DataType, Properties...> const& view);
   ```

 * ```c++
    resize(const size_t n0 = 0, const size_t n1 = 0, const size_t n2 = 0, const size_t n3 = 0, const size_t n4 = 0, const size_t n5 = 0, const size_t n6 = 0, const size_t n7 = 0);
   ```
   Resize a view with copying old data to new data at the corresponding indices.

 * ```c++
    realloc(const size_t n0 = 0, const size_t n1 = 0, const size_t n2 = 0, const size_t n3 = 0, const size_t n4 = 0, const size_t n5 = 0, const size_t n6 = 0, const size_t n7 = 0);
   ```
   Resize a view with discarding old data.

### Free Functions
 * ```c++
   contribute(View<DataType1, Properties...>& dest, Kokkos::Experimental::ScatterView<DataType2, Layout, ExecSpace, Operation, Contribution, Duplicated> const& src)
   ```
   Convenience function to perform final reduction of ScatterView results into a resultant View; may be called following [`parallel_reduce()`](../core/parallel-dispatch/parallel_reduce).
