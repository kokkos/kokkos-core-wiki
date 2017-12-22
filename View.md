# Chapter 6

# View: Multidimensional array

After reading this chapter, you should understand the following:

* A Kokkos View is an array of zero or more dimensions
* How to use View's first template parameter to specify the type of entries, the number of dimensions, and whether the dimensions are determined at run time or compile time
* Kokkos handles array deallocation automatically
* Kokkos chooses array layout at compile time for best overall performance as a function of the computer architecture
* How to set optional template parameters of View for low-level control of execution space, layout, and how Kokkos accesses array entries

In all code examples in this chapter, we assume that all classes in the `Kokkos` namespace have been imported into the working namespace.


## 6.1 Why Kokkos needs multidimensional arrays

Many scientific and engineering codes spend a lot of time computing with arrays of data and programmers invest a lot of effort making these array computations as fast as possible. This effort is often intimately bound to details of the computer architecture, run-time environment, language, and programming model. For example, optimal array layout may differ based on the architecture, with a large integer factor penalty if wrong. Low-level issues like pointer alignment, array layout, indexing overhead, and initialization all affect performance. This is true even for sequential codes but thread parallelism adds even more pitfalls, like first-touch allocation and false sharing.

For best performance, coders need to tie details of how they manage arrays to details of how parallel code accesses and manages those arrays. Programmers who write architecture-specific code then need to mix low-level features of the architecture and programming model into high-level algorithms. This makes it hard to port codes between architectures, or to write a code that still performs well as architectures evolve.

Kokkos aims to relieve some of this burden by optimizing array management and access for the specific architecture. Tying arrays to shared-memory parallelism lets Kokkos optimize the former to the latter. For example, Kokkos can easily do first-touch allocation because it controls threads that it can use to initialize arrays. Kokkos' architecture-awareness lets it pick optimal layout and pad allocations for good alignment. Expert coders can also use Kokkos to access low-level or more architecture-specific optimizations in a more user-friendly way. For instance, Kokkos makes it easy to experiment with different array layouts.


## 6.2 Creating and using a View

### 6.2.1 Constructing a View

A View is an array of zero or more dimensions. Programmers set both the type of entries and the number of dimensions at compile time as part of the type of the View. For example, the following specifies and allocates a View with four dimensions for entries that have type `int`:

```c++
const size_t N0 = ...;
const size_t N1 = ...;
const size_t N2 = ...;
const size_t N3 = ...;
View<int****> a ("some label", N0, N1, N2, N3);
```
The string argument is a label which Kokkos uses for debugging. Different Views may have the same label. The ellipses indicate some integer dimensions specified at run time. Users may also set some dimensions at compile time. For example, the following View has two dimensions where the first (represented by the asterisk) is a run-time dimension and the second (represented by [3]) is a compile-time dimension. Thus, the View is an N by 3 array of type double, where N is specified at run time in the View's constructor.
```c++
const size_t N = ...;
View<double*[3]> b ("another label", N);
```
Views may have up to (at most) 8 dimensions and any number of these may be run-time or compile-time. The only limitation is that the run-time dimensions (if any) must appear first followed by all the compile-time dimensions (if any). For example, the following are valid three-dimensional View types:

* `View<int***>`  (3 run-time dimensions)
* `View<int**[8]>`  (2 run-time, 1 compile-time)
* `View<int*[3][8]>`  (1 run-time, 2 compile-time)
* `View<int[4][3][8]>`  (3 compile-time)

and the following are _not_ valid three-dimensional View types:

* `View<int[4]**>`
* `View<int[4][3]*>`
* `View<int[4]*[8]>`
* `View<int*[3]*>`

This limitation comes from the implementation of View using C++ templates. View's first template parameter must be a valid C++ type.

Note that the above used constructor is not necessarily available for all view types; specific Layouts or Memory Spaces may require more specialized allocators. This is discussed later.

Another important thing to keep in mind is that a `View` handle is a stateful object. It is not legal to create a `View` handle from raw memory by typecasting a pointer. To call any operator on a `View,` including the assignment operator, its constructor must have been called before. If it is necessary to initialize raw memory with a `View` handle, one can legally do so using a move constructor ("placement new"). The above has nothing to do with the data a `View` is referencing. It is completely legal to give a typecast pointer to the constructor of an unmanaged `View`.

```c++
View<int*> *a_ptr = (View<int*>*) malloc(10*sizeof(View<int*);
a_ptr[0] = View<int*>("A0",1000); // This is illegal
new(&a_ptr[1]) View<int*>("A1",10000); // This is legal 
```

### 6.2.2 What types of data may a View contain?

C++ lets users construct data types that may "look like" numbers in terms of syntax but do arbitrarily complicated things inside. Some of those things may not be thread safe, like unprotected updates to global state. Others may perform badly in parallel, like fine-grained dynamic memory allocation. Therefore, it is strongly advised to use only simple data types inside Kokkos Views. Users may always construct a View whose entries are

* built-in data types ("plain old data"), like `int` or `double`, or
* structs of built-in data types.

While it is in principle possible to have Kokkos Views of arbitrary objects, Kokkos imposes restrictions on the set of types `T` for which one can construct a `View<T*>`.  For example:

* `T` must have a default constructor and destructor. 
* `T` must not have virtual methods.
* `T`'s default constructor and destructor must not allocate data, and must be thread safe. 
* `T`'s assignment operators as well as its default constructor and deconstructor must be marked with the `KOKKOS_INLINE_FUNCTION` or `KOKKOS_FUNCTION` macro.

All restrictions but the first come from the requirement that `View<T*>` work with every execution and memory space,
including those that use CUDA. The constructor of `View<T*>` does not just allocate memory; it also initializes the allocation by iterating over it using a `Kokkos::parallel_for`, and invoking `T`'s default constructor for each entry. If the View's execution space is `Cuda`, then that means `T`'s default constructor needs to be correct to call on device. Keep in mind that the semantics of the resulting `View` are a combination of the `Views 'view'` semantics and the behavior of the element type.

Finally, note that virtual functions are technically allowed, but calling them is subject to further restrictions; developers should consult the discussions in Chapter 13, Kokkos and Virtual Functions (under development).

### 6.2.3 Const Views

A view can have const data semantics (i.e. its entries are read-only) by specifying a `const` data type.  It is a compile-time error to assign to an entry of a "const View". Assignment semantics are equivalent to a pointer to const data. A const View means the _entries_ are const; you may still assign to a const View. `View<const double*>` corresponds exactly to `const double*`, and `const View<double*>` to `double* const`. Therefore, it does not make sense to allocate a const View since you could not obtain a non-const view of the same data and you can not assign to it. You can however assign a non-const view to a const view. Here is an example:
```c++
const size_t N0 = ...;
View<double*> a_nonconst ("a_nonconst", N0);

// Assign a nonconst View to a const View
View<const double*> a_const = a_nonconst;
// Pass the const View to some read-only function.
const double result = readOnlyFunction (a_const);
```

Const Views often enables the compiler to optimize more aggressively by allowing it to reason about possible write conflicts and data aliasing. For example, in a vector update `a(i+1)+=b(i)` with skewed indexing, it is safe to vectorize if `b` is a View of const data.

### 6.2.4 Accessing entries (indexing)

You may access an entry of a View using parentheses enclosing a comma-delimited list of integer indices. This looks just like a Fortran multidimensional array access. For example:

```c++
const size_t N = ...;
View<double*[3][4]> a ("some label", N);
// KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
parallel_for (N, KOKKOS_LAMBDA (const ptrdiff_t i) {
  const size_t j = ...;
  const size_t k = ...;
  const double a_ijk = a(i,j,k);
  /* rest of the loop body */
});
```

Note how in the above example, we only access the View's entries in a parallel loop body. In general, you may only access a View's entries in an execution space which is allowed to access that View's memory space. For example, if the default execution space is `Cuda`, a View for which no specific Memory Space was given may not be accessed in host code<sup>1</sup>.

Furthermore, access costs (e.g., latency and bandwidth) may vary, depending on the View's "native" memory and execution spaces and the execution space from which you access it. CUDA UVM may work, but it may also be slow, depending on your access pattern and performance requirements. Thus, best practice is to access the View only in a Kokkos parallel for, reduce, or scan, using the same execution space as the View. This also ensures that access to the View's entries respect first-touch allocation. The first (leftmost) dimension of the View is the _parallel dimension_ over which it is most efficient to do parallel array access if the default memory layout is used (e.g. if no specific memory layout is
specified).

***
<sup>1</sup>  An exemption is if you specified for CUDA compilation that the default memory space is CudaUVMSpace, which can be accessed from the host.

***

### 6.2.5 Reference counting

Kokkos automatically manages deallocation of Views through a reference-counting mechanism.  Otherwise, Views behave like raw pointers. Copying or assigning a View does a shallow copy, and changes the reference count. (The View copied has its reference count incremented, and the assigned-to View has its reference count decremented.) A View's destructor (called when the View falls out of scope or during a stack unwind due to an exception) decrements the reference count. Once the reference count reaches zero, Kokkos may deallocate the View.

For example, the following code allocates two Views, then assigns one to the other. That assignment may deallocate the first View, since it reduces its reference count to zero. It then increases the reference count of the second View, since now both Views point to it.

```c++
View<int*> a ("a", 10);
View<int*> b ("b", 10);
a = b; // assignment does shallow copy
```

For efficiency, View allocation and reference counting turn off inside of Kokkos' parallel for, reduce, and scan operations. This affects what you can do with Views inside of Kokkos' parallel operations.

### 6.2.6 Resizing

Kokkos Views can be resized using the `resize` non-member function. It takes an existing view as its input by reference and the new dimension information corresponding to the constructor arguments. A new view with the new dimensions will be created and a kernel will be run in the views execution space to copy the data element by element from the old view to the new one. Note that the old allocation is only deleted if the view to be resized was the _only_ view referencing the underlying allocation.

```c++
// Allocate a view with 100x50x4 elements
View<int**[4]> a( "a", 100,50);
    
// Resize a to 200x50x4 elements; the original allocation is freed
resize(a, 200,50);
    
// Create a second view b viewing the same data as a
View<int**[4]> b = a;
// Resize a again to 300x60x4 elements; b is still 200x50x4
resize(a,300,60);
```

## 6.3 Layout

### 6.3.1 Strides and dimensions

_Layout_ refers to the mapping from a logical multidimensional index _(i, j, k, . . .)_ to a physical memory offset. Different programming languages may have different layout conventions. For example, Fortran uses _column-major_ or "left" layout, where consecutive entries in the same column of a 2-D array are contiguous in memory. Kokkos calls this `LayoutLeft`. C, C++, and Java use _row-major_ or "right" layout, where consecutive entries in the same row of a 2-D array are contiguous in memory. Kokkos calls this `LayoutRight`.

The generalization of both left and right layouts is "strided." For a strided layout, each dimension has a _stride_. The stride for that dimension determines how far apart in memory two array entries are, whose indices in that dimension differ only by one, and whose other indices are all the same. For example, with a 3-D strided view with strides _(s_1, s_2, s_3)_, entries _(i, j, k)_ and _(i, j+1, k)_ are _s_2_ entries (not bytes) apart in memory. Kokkos calls this `LayoutStride`.

Strides may differ from dimensions. For example, Kokkos reserves the right to pad each dimension for cache or vector alignment. You may access the dimensions of a View using the `dimension` method, which takes the index of the dimension.

Strides are accessed using the `stride` method. It takes a raw integer array, and only fills in as many entries as the rank of the View. For example:

```c++
const size_t N0 = ...;
const size_t N1 = ...;
const size_t N2 = ...;
View<int***> a ("a", N0, N1, N2);
    
int dim1 = a.dimension (1); // returns dimension 1
size_t strides[3]
a.strides (dims); // fill 'strides' with strides
```

You may also refer to specific dimensions without a runtime parameter:

```c++
const size_t n0 = a.dimension_0 ();
const size_t n2 = a.dimension_2 ();
```

Note the return type of `dimension_N()` is the `size_type` of the views memory space. This causes some issues if warning-free compilation should be achieved since it will typically be necessary to cast the return value. In particular, in cases where the `size_type` is more conservative than required, it can be beneficial to cast the value to `int` since signed 32 bit integers typically give the best performance when used as index types. In index heavy codes, this performance difference can be significant compared to using `size_t` since the vector length on many architectures is twice as long for 32 bit values as for 64 bit values and signed integers have less stringent overflow testing requirements than unsigned integers.

Users of the BLAS and LAPACK libraries may be familiar with the ideas of layout and stride. These libraries only accept matrices in column-major format. The stride between consecutive entries in the same column is 1, and the stride between consecutive entries in the same row is `LDA` ("leading dimension of the matrix A"). The number of rows may be less than `LDA`, but may not be greater.

### 6.3.2 Other layouts

Other layouts are possible.  For example, Kokkos has a "tiled" layout, where a tile's entries are stored contiguously (in either row- or column-major order) and tiles have compile-time dimensions. One may also use Kokkos to implement Morton ordering or variants thereof. In order to write a custom layout one has to define a new layout class and specialise the `ViewMapping` class for that layout. The `ViewMapping` class implements the offset operator as well as stride calculation for regular layouts. A good way to start such a customization is by copying the implementation of `LayoutLeft` and its associated `ViewMapping` specialization, renaming the layout and then change the offset operator.

### 6.3.3 Default layout depends on execution space

Kokkos selects a View's default layout for optimal parallel access over the leftmost dimension based on its execution space. For example, `View<int**, Cuda>` has `LayoutLeft`, so that consecutive threads in the same warp access consecutive entries in memory. This _coalesced access_ gives the code better memory bandwidth.

In contrast, `View<int**, OpenMP>` has `LayoutRight`, so that a single thread accesses contiguous entries of the array. This avoids wasting cache lines and helps prevent false sharing of a cache line between threads. In Section 6.4 more details will be discussed.

### 6.3.4 Explicitly specifying layout

We prefer that users let Kokkos determine a View's layout, based on its execution space. However, sometimes you really need to specify the layout. For example, the BLAS and LAPACK libraries only accept column-major arrays.  If you want to give a View to the BLAS or LAPACK library, that View must be `LayoutLeft`. You may specify the layout as a template parameter of View. For example:

```c++
const size_t N0 = ...;
const size_t N1 = ...;
View<double**, LayoutLeft> A ("A", N0, N1);
    
// Get 'LDA' for BLAS / LAPACK
int strides[2]; // any integer type works in stride()
A.stride (strides);
const int LDA = strides[1];
```

You may ask a View for its layout via its `array_layout` typedef. This can be helpful for C++ template metaprogramming. For example:

```c++
template<class ViewType>
void callBlas (const ViewType& A) {
  typedef typename ViewType::array_layout array_layout;
  if (std::is_same<array_layout, LayoutLeft>::value) {
    callSomeBlasFunction (A.data(), ...);
  } else {
    throw std::invalid_argument ("A is not LayoutLeft");
  }
}
```

## 6.4 Managing Data Placement

### 6.4.1 Memory spaces

Views are allocated by default in the default execution space's default memory space. You may access the View's execution space via its `execution_space` typedef, and its memory space via its `memory_space` typedef. You may also specify the memory space explicitly as a template parameter. For example, the following allocates a View in CUDA device memory:

```c++
View<int*, CudaSpace> a ("a", 100000);
```

and the following allocates a View in "host" memory, using the default host execution space for first-touch initialization:

```c++
View<int*, HostSpace> a ("a", 100000);
```

Since there is no bijective association between execution spaces and memory spaces, Kokkos provides a way to explicitly provide both to a View as a `Device`.

```c++
View<int*, Device<Cuda,CudaUVMSpace> > a ("a", 100000);
View<int*, Device<OpenMP,CudaUVMSpace> > b ("b", 100000);
```

In this case `a` and `b` will live in the same memory space, but `a` will be initialized on the GPU while `b` will be
initialized on the host. The `Device` type can be accessed as a views `device_type` typedef. A `Device` has only three typedef members: `device_type`, `execution_space` and `memory_space`. The `execution_space` and `memory_space` typedefs are the same for a view and the `device_type` typedef.

It is important to understand that accessibility of a View does not depend on its execution space directly. It is only determined by its memory space. Therefore both `a` and `b` have the same access properties. They differ only in how they are initialized and in where parallel kernels associated with operations such as resizing or deep copies are run.

### 6.4.2 Initialization

A View's entries are initialized to zero by default. Initialization happens in parallel for first-touch allocation over the first (leftmost) dimension of the View using the execution space of the View.

You may allocate a View without initializing. For example:

```c++
View<int*> x (ViewAllocateWithoutInitializing (label), 100000);
```

This is useful in situations where your dominant use of the View exhibits a complicated access pattern. In this case it is best to run the most costly kernel directly after initialization to execute the first touch pattern and get optimal memory affinity.

### 6.4.3 Deep copy and HostMirror

Copying data from one view to another, in particular between views in different memory spaces, is called deep copy.
Kokkos never performs a hidden deep copy. To do so a user has to call the `deep_copy` function. For example:

```c++
View<int*> a ("a", 10);
View<int*> b ("b", 10);
deep_copy (a, b); // copy contents of b into a
```

Deep copies can only be performed between views with an identical memory layout and padding. For example the following two operations are not valid:

```c++
View<int*[3], CudaSpace> a ("a", 10);
View<int*[3], HostSpace> b ("b", 10);
deep_copy (a, b); // This will give a compiler error

View<int*[3], LayoutLeft, CudaSpace> c ("c", 10);
View<int*[3], LayoutLeft, HostSpace> d ("d", 10);
deep_copy (c, d); // This might give a runtime error
```

The first one will not work because the default layouts of `CudaSpace` and `HostSpace` are different. The compiler will catch that since no overload of the `deep_copy` function exists to copy view from one layout to another. The second case will fail at runtime if padding settings are different for the two memory spaces. This would result in different allocation sizes and thus prevent a direct memcopy.

The reasoning for allowing only direct bitwise copies is that a deep copy between different memory spaces would otherwise require a temporary copy of the data to which a bitwise copy is performed followed by a parallel kernel to transfer the data element by element.

Kokkos provides the following way to work around those limitations. First views have a `HostMirror` typedef which is a view type with compatible layout inside the `HostSpace`. Additionally there is a `create_mirror` and `create_mirror_view` function which allocate views of the `HostMirror` type of a view. The difference between the two is that `create_mirror` will always allocate a new view, while `create_mirror_view` will only create a new view if the original one is not in `HostSpace`.

```c++
View<int*[3], MemorySpace> a ("a", 10);
// Allocate a view in HostSpace with the layout and padding of a
typename View<int*[3], MemorySpace>::HostMirror b =
    create_mirror(a);
// This is always a memcopy
deep_copy (a, b);
    
typename View<int*[3]>::HostMirror c =
create_mirror_view(a);
// This is a no-op if MemorySpace is HostSpace
deep_copy (a, c)
```

### 6.4.4 How do I get the raw pointer?

We discourage access to a View's "raw" pointer. This circumvents reference counting, that is, the memory may be deallocated once the View's reference count goes to zero so holding on to a raw pointer may result in invalid memory access. Furthermore, it may not even be possible to access the View's memory from a given execution space. For example, a View in the `Cuda` space points to CUDA device memory. Also using raw pointers would normally defeat the usability of polymorpic layouts and automatic padding. Nevertheless, for instances you really need access to the pointer, we provide the `data()` method. For example:

```c++
// Legacy function that takes a raw pointer.
extern void legacyFunction (double* x_raw, const size_t len);
  
// Your function that takes a View.
void myFunction (const View<double*>& x) {
  // DON'T DO THIS UNLESS YOU MUST
  double* a_raw = a.data ();
  const size_t N = x.dimension_0 ();
  legacyFunction (a_raw, N);
}
```

A user is in most cases also allowed to obtain a pointer to a specific element via the usual `&` operator. For example

```c++
// Legacy function that takes a raw pointer.
void someLibraryFunction (double* x_raw);
    
KOKKOS_INLINE_FUNCTION
void foo(const View<double*>& x) {
  someLibraryFunction(&x(3));
}
```

This is only valid if a Views reference type is an `lvalue`. That property can be queried statically at compile time from the view through its `reference_is_lvalue` member.


## 6.5 Access traits

Another way to get optimized data accesses is to specify a memory trait. These traits are used to declare intended use of the particular view of an allocation. For example, a particular kernel might use a view only for streaming writes. By declaring that intention, Kokkos can insert the appropriate store intrinsics on each architecture if available. Access traits are specified through an optional template parameter which comes last in the list of parameters. Multiple traits can be combined with binary "or" operators:

```c++
View<double*, MemoryTraits<SomeTrait> > a;
View<const double*, MemoryTraits<SomeTrait | SomeOtherTrait> > b;
View<int*, LayoutLeft, MemoryTraits<SomeTrait | SomeOtherTrait> > c;
View<int*, MemorySpace, MemoryTraits<SomeTrait | SomeOtherTrait> > d;
View<int*, LayoutLeft, MemorySpace, MemoryTraits<SomeTrait> > e;
```

### 6.5.1 Atomic access

The `Atomic` access trait lets you create a View of data such that every read or write to any entry uses an atomic update. Kokkos supports atomics for all data types independent of size. Restrictions are that you are
1. not allowed to alias data for which atomic operations are performed, and 
1. the results of non atomic accesses (including read) to data which is at the same time atomically accessed is not defined.

Performance characteristics of atomic operations depend on the data type. Some types (in particular integer types) are natively supported and might even provide asynchronous atomic operations. Others (such as 32 bit and 64 bit atomics for non-integer types) are often implemented using CAS loops of integers. Everything else is implemented with a locking approach where an atomic operation acquires a lock based on a hash of the pointer value of the data element.

Types for which atomic access are performed must support the necessary operators such as =, +=, -=, +, - etc. as well as have a number of `volatile` overloads of functions such as assign and copy constructors defined. 

```c++
View<int*> a("a" , 100);
View<int*, MemoryTraits<Atomic> > a_atomic = a;
    
a_atomic(1) += 1; // This access will do an atomic addition
```

### 6.5.2 Random Access

The `RandomAccess` trait declares the intent to access a View irregularly (in particular non consecutively). If used for a const View in the `CudaSpace` or `CudaUVMSpace`, Kokkos will use texture fetches for accesses when executing in the `Cuda` execution space. For example:

```c++
const size_t N0 = ...;
View<int*> a_nonconst ("a", N0); // allocate nonconst View
// Assign to const, RandomAccess View
View<const int*, RandomAccess> a_ra = a_nonconst;
```

If the default execution space is `Cuda`, access to a `RandomAccess` View may use CUDA texture fetches. Texture fetches are not cache coherent with respect to writes so you must use read-only access. The texture cache is optimized for noncontiguous access since it has a shorter cache line than the regular cache.

While `RandomAccess` is valid for other execution spaces, currently no specific optimizations are performed. But in the future a view allocated with the `RandomAccess` attribute might for example, use a larger page size, and thus reduce page faults in the memory system.

### 6.5.3 Standard idiom for specifying access traits

The standard idiom for View is to pass it around using as few template parameters as possible. Then, assign to a View with the desired access traits only at the "last moment" when those access traits are needed just before entering a computational kernel. This lets you template C++ classes and functions on the View type without proliferating instantiations. Here is an example:

```c++
// Compute a sparse matrix-vector product, for a sparse
// matrix stored in compressed sparse row (CSR) format.
void spmatvec (const View<double*>& y,
      const View<const size_t*>& ptr,
      const View<const int*>& ind,
      const View<const double*>& val,
      const View<const double*>& x)
{
  // Access to x has less locality than access to y.
  View<const double*, RandomAccess> x_ra = x;
  typedef View<const size_t*>::size_type size_type;
    
  parallel_for (y.dimension_0 (), KOKKOS_LAMBDA (const size_type i) {
    double y_i = y(i);
    for (size_t k = ptr(i); k < ptr(i+1); ++k) {
      y_i += val(k) * x_ra(ind(k));
    }
    y(i) = y_i;
  });
}
```

### 6.5.4 Unmanaged Views

It's always better to let Kokkos control memory allocation but sometimes you don't have a choice. You might have to work with an application or an interface that returns a raw pointer, for example. Kokkos lets you wrap raw pointers in an _unmanaged View_. "Unmanaged" means that Kokkos does _not_ do reference counting or automatic deallocation for those Views. The following example shows how to create an unmanaged View of host memory. You may do this for CUDA device memory too, or indeed for memory allocated in any memory space, by specifying the View's execution or memory space accordingly.

```c++
// Sometimes other code gives you a raw pointer, ...
const size_t N0 = ...;
double* x_raw = malloc (N0 * sizeof (double));
{
  // ... but you want to access it with Kokkos.
  //
  // malloc() returns host memory, so we use the host memory space HostSpace.  
  // Unmanaged Views have no label because labels work with the reference counting system.
  View<double*, HostSpace, MemoryTraits<Unmanaged> >
    x_view (x_raw, N0);
 
  functionThatTakesKokkosView (x_view);
  
  // It's safest for unmanaged Views to fall out of scope before freeing their memory.
}
free (x_raw);
```

### 6.5.5 Conversion Rules and Function Specialization

Not all view types can be assigned to each other. Requirements are: 
* the data type and dimension have to match, 
* the layout must be compatible and 
* the memory space has to match.

Examples illustrating the rules are:

1.  Data Type and Rank has to Match
    *  `int*  -> int*          `   / ok
    *  `int*  -> const int*    `   / ok
    *  `const int*  -> int*    `   / not ok, const violation
    *  `int** -> int*          `   / not ok, rank mismatch
    *  `int*[3] -> int**       `   / ok
    *  `int** -> int*[3]       `   / ok if runtime dimension check matches
    *  `int*  -> long*         `   / not ok, type mismatch
2.  Layouts must be compatible
    *  `LayoutRight -> LayoutRight `  / ok
    *  `LayoutLeft -> LayoutRight  `  / not ok except for 1D Views
    *  `LayoutLeft -> LayoutSride  `  / ok
    *  `LayoutStride -> LayoutLeft `  / ok if runtime dimensions allow assignment
3.  Memory Spaces must match
    *  `Kokkos::View<int*> -> Kokkos::View<int*,HostSpace>`  / ok if default memory space is HostSpace
4.  Memory Traits

**[[Chapter 7: Parallel Dispatch|ParallelDispatch]]**