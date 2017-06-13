# Chapter

# 6 View: Multidimensional array

After reading this chapter, you should understand the following:

* A Kokkos View is an array of zero or more dimensions
* How to use View's first template parameter to specify the type of entries, the number of dimensions, and whether the dimensions are determined at run time or compile time
* Kokkos handles array deallocation automatically
* Kokkos chooses array layout at compile time for best overall performance, as a function of the computer architecture
* How to set optional template parameters of View for low-level control of execution space, layout, and how Kokkos accesses array entries

In all code examples in this chapter, we assume that all classes in the `Kokkos` namespace have been imported into the working namespace.


## 6.1 Why Kokkos needs multidimensional arrays

Many scientific and engineering codes spend a lot of time computing with arrays of data. Programmers invest a lot of effort making array computations as fast as possible. This effort is often intimately bound to details of the computer architecture, run-time environment, language, and programming model. For example, optimal array layout may differ based on the architecture, with a large integer factor penalty if wrong. Low-level issues like pointer alignment, array layout, indexing overhead, and initialization all affect performance. This is true even with sequential codes. Thread parallelism adds even more pitfalls, like first-touch allocation and false sharing.

For best performance, coders need to tie details of how they manage arrays to details of how parallel code accesses and manages those arrays. Programmers who write architecture-specific code then need to mix low-level features of the architecture and programming model into high-level algorithms. This makes it hard to port codes between architectures, or to write a code that still performs well as architectures evolve.

Kokkos aims to relieve some of this burden by optimizing array management and access for the specific architecture. Tying arrays to shared-memory parallelism lets Kokkos optimize the former to the latter. For example, Kokkos can easily do first-touch allocation because it controls threads that it can use to initialize arrays. Kokkos' architecture awareness lets it pick optimal layout and pad allocations for good alignment. Expert coders can also use Kokkos to access low-level or more architecture-specific optimizations in a more user-friendly way. For instance, Kokkos makes it easy to experiment with different array layouts.


## 6.2 Creating and using a View

### 6.2.1 Constructing a View

A View is an array of zero or more dimensions. Programmers set both the type of entries, and the number of dimensions, at compile time, as part of the type of the View. For example, the following specifies and allocates a View with four dimensions, whose entries have type `int`:

    const size_t N0 = ...;
    const size_t N1 = ...;
    const size_t N2 = ...;
    const size_t N3 = ...;
    View<int****> a ("some label", N0, N1, N2, N3);

The string argument is a label which Kokkos uses for debugging. Different Views may have the same label. The ellipses indicate some integer dimensions specified at run time. Users may also set some dimensions at compile time. For example, the following View has two dimensions. The first one (represented by the asterisk) is a run-time dimension, and the second (represented by [3]) is a compile-time dimension. Thus, the View is an N by 3 array of type double, where N
is specified at run time in the View's constructor.

    const size_t N = ...;
    View<double*[3]> b ("another label", N);

Views may have up to (at most) 8 dimensions, and any number of these may be run-time or compile-time. The only limitation is that the run-time dimensions (if any) must appear first, followed by all the compile-time dimensions (if any). For example, the following are valid three-dimensional View types:

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

    View<int*> *a_ptr = (View<int*>*) malloc(10*sizeof(View<int*);
    a_ptr[0] = View<int*>("A0",1000); // This is illegal
    new(&a_ptr[1]) View<int*>("A1",10000); // This is legal 


### 6.2.2 What types of data may a View contain?

C++ lets users construct data types that may "look like" numbers in terms of syntax but do arbitrarily complicated things inside. Some of those things may not be thread safe, like unprotected updates to global state. Others may perform badly in parallel, like fine-grained dynamic memory allocation. Therefore, it is strongly advised to use only simple data types inside Kokkos Views. Users may always construct a View whose entries are

* built-in data types (``plain old data''), like `int or double`, or
* structs of built-in data types.

While it is in principal possible to have Kokkos Views of arbitrary objects, there are a number of restrictions. The `T` type must have a default constructor and destructor. For portability reasons, `T` is not allowed to have virtual functions and one is not allowed to call functions with dynamic memory allocations inside of kernel code. Furthermore, assignment operators as well as default constructor and deconstructor must be marked with `KOKKOS_[INLINE_]FUNCTION`. Keep in mind that the semantics of the resulting View are a combination of the Views 'view' semantics and the behaviour of the element type.

### 6.2.3 Const Views

A view can have const data semantics (i.e. its entries are read-only) by specifying a `const` data type.  It is a compile-time error to assign to an entry of a "const View". Assignment semantics are equivalent to a pointer to const data. A const View means the _entries_ are const; you may still assign to a const View. `View<const double*>` corresponds exactly to `const double*`, and `const View<double*>` to `double* const`. Therefore, it does not make sense to allocate a const View since you could not obtain a non-const view of the same data and you can not assign to it. You can however assign a non-const view to a const view. Here is an example:

    const size_t N0 = ...;
    View<double*> a_nonconst ("a_nonconst", N0);

    // Assign a nonconst View to a const View
    View<const double*> a_const = a_nonconst;
    // Pass the const View to some read-only function.
    const double result = readOnlyFunction (a_const);


Const Views often enables the compiler to optimize more aggressively by allowing it to reason about possible write conflicts and data aliasing. For example, in a vector update `a(i+1)+=b(i)` with skewed indexing, it is safe to vectorize if `b` is a View of const data.

### 6.2.4 Accessing entries (indexing)

You may access an entry of a View using parentheses enclosing a comma-delimited list of integer indices. This looks just like a Fortran multidimensional array access. For example:

    const size_t N = ...;
    View<double*[3][4]> a ("some label", N);
    // KOKKOS_LAMBDA macro includes capture-by-value specifier [=].
    parallel_for (N, KOKKOS_LAMBDA (const ptrdiff_t i) {
        const size_t j = ...;
        const size_t k = ...;
        const double a_ijk = a(i,j,k);
        /* rest of the loop body */
    });

Note how in the above example, we only access the View's entries in a parallel loop body. In general, you may only access a View's entries in an execution space which is allowed to access that View's memory space. For example, if the default execution space is `Cuda`, a View for which no specific Memory Space was given may not be accessed in host code 
\footnote{An exemption is if you specified for CUDA compilation that the default memory space is CudaUVMSpace, which can be accessed from the host.}.

Furthermore, access costs (e.g., latency and bandwidth) may vary, depending on the View's "native" memory and execution spaces and the execution space from which you access it. CUDA UVM may work, but it may also be slow, depending on your access pattern and performance requirements. Thus, best practice is to access the View only in a Kokkos parallel for, reduce, or scan, using the same execution space as the View. This also ensures that access to the View's entries respect first-touch allocation. The first (leftmost) dimension of the View is the _parallel dimension_ over which it is most efficient to do parallel array access if the default memory layout is used (e.g. if no specific memory layout is
specified).

### 6.2.5 Reference counting

Kokkos automatically manages deallocation of Views through a reference-counting mechanism.  Otherwise, Views behave like raw pointers. Copying or assigning a View does a shallow copy, and changes the reference count. (The View copied has its reference count incremented, and the assigned-to View has its reference count decremented.) A View's destructor (called when the View falls out of scope or during a stack unwind due to an exception) decrements the reference count. Once the reference count reaches zero, Kokkos may deallocate the View.

For example, the following code allocates two Views, then assigns one to the other. That assignment may deallocate the first View, since it reduces its reference count to zero. It then increases the reference count of the second View, since now both Views point to it.

    View<int*> a ("a", 10);
    View<int*> b ("b", 10);
    a = b; // assignment does shallow copy


For efficiency, View allocation and reference counting turn off inside of Kokkos' parallel for, reduce, and scan operations. This affects what you can do with Views inside of Kokkos' parallel operations.

### 6.2.6 Resizing

Kokkos Views can be resized using the `resize` non-member function. It takes an existing view as its input by reference and the new dimension information corresponding to the constructor arguments. A new view with the new dimensions will be created and a kernel will be run in the views execution space to copy the data element by element from the old view to the new one. Note that the old allocation is only deleted if the view to be resized was the _only_ view referencing the underlying allocation.

    // Allocate a view with 100x50x4 elements
    View<int**[4]> a( "a", 100,50);
    
    // Resize a to 200x50x4 elements; the original allocation is freed
    resize(a, 200,50);
    
    // Create a second view b viewing the same data as a
    View<int**[4]> b = a;
    // Resize a again to 300x60x4 elements; b is still 200x50x4
    resize(a,300,60);


## 6.3 Layout

### 6.3.1 Strides and dimensions

_Layout_ refers to the mapping from a logical multidimensional index _(i, j, k, . . .)_ to a physical memory offset. Different programming languages may have different layout conventions. For example, Fortran uses _column-major_ or "left" layout, where consecutive entries in the same column of a 2-D array are contiguous in memory. Kokkos calls this `LayoutLeft`. C, C++, and Java use _row-major_ or "right" layout, where consecutive entries in the same row of a 2-D array are contiguous in memory. Kokkos calls this `LayoutRight`.

The generalization of both left and right layouts is "strided." For a strided layout, each dimension has a _stride_. The stride for that dimension determines how far apart in memory two array entries are, whose indices in that dimension differ only by one, and whose other indices are all the same. For example, with a 3-D strided view with strides _(s_1, s_2, s_3)_, entries _(i, j, k)_ and _(i, j+1, k)_ are _s_2_ entries (not bytes) apart in memory. Kokkos calls this `LayoutStride`.

Strides may differ from dimensions. For example, Kokkos reserves the right to pad each dimension for cache or vector alignment. You may access the dimensions of a View using the `dimension` method, which takes the index of the dimension.

Strides are accessed using the `stride` method. It takes a raw integer array, and only fills in as many entries as the rank of the View. For example:

    const size_t N0 = ...;
    const size_t N1 = ...;
    const size_t N2 = ...;
    View<int***> a ("a", N0, N1, N2);
    
    int dim1 = a.dimension (1); // returns dimension 1
    size_t strides[3]
    a.strides (dims); // fill 'strides' with strides

You may also refer to specific dimensions without a runtime parameter:

    const size_t n0 = a.dimension_0 ();
    const size_t n2 = a.dimension_2 ();

Note the return type of `dimension_N()` is the `size_type` of the views memory space. This causes some issues if warning free compilation should be achieved since it will typically be necessary to cast the return value. In particular, in cases where the `size_type` is more conservative than required, it can be beneficial to cast the value to `int` since signed 32 bit integers typically give the best performance when used as index types. In index heavy codes, this performance difference can be significant compared to using `size_t` since the vector length on many architectures is twice as long for 32 bit values as for 64 bit values and signed integers have less stringent overflow testing requirements than unsigned integers.

Users of the BLAS and LAPACK libraries may be familiar with the ideas of layout and stride. These libraries only accept matrices in column-major format. The stride between consecutive entries in the same column is 1, and the stride between consecutive entries in the same row is `LDA` (``leading dimension of the matrix A''). The number of rows may be less than `LDA`, but may not be greater.

### 6.3.2 Other layouts

Other layouts are possible.  For example, Kokkos has a "tiled" layout, where a tile's entries are stored contiguously (in either row- or column-major order) and tiles have compile-time dimensions. One may also use Kokkos to implement Morton ordering or variants thereof. In order to write a custom layout one has to define a new layout class and specialise the `ViewMapping` class for that layout. The `ViewMapping` class implements the offset operator as well as stride calculation for regular layouts. A good way to start such a customization is by copying the implementation of `LayoutLeft` and its associated `ViewMapping` specialization, renaming the layout and then change the offset operator.

### 6.3.3 Default layout depends on execution space

Kokkos selects a View's default layout for optimal parallel access over the leftmost dimension based on its execution space. For example, `View<int**, Cuda>` has `LayoutLeft`, so that consecutive threads in the same warp access consecutive entries in memory. This _coalesced access_ gives the code better memory bandwidth.

In contrast, `View<int**, OpenMP>` has `LayoutRight`, so that a single thread accesses contiguous entries of the array. This avoids wasting cache lines and helps prevent false sharing of a cache line between threads. In Section 6.4 more details will be discussed.

### 6.3.4 Explicitly specifying layout

We prefer that users let Kokkos determine a View's layout, based on its execution space. However, sometimes you really need to specify the layout. For example, the BLAS and LAPACK libraries only accept column-major arrays.  If you want to give a View to the BLAS or LAPACK library, that View must be `LayoutLeft`. You may specify the layout as a template parameter of View. For example:

    const size_t N0 = ...;
    const size_t N1 = ...;
    View<double**, LayoutLeft> A ("A", N0, N1);
    
    // Get 'LDA' for BLAS / LAPACK
    int strides[2]; // any integer type works in stride()
    A.stride (strides);
    const int LDA = strides[1];

You may ask a View for its layout via its `array_layout` typedef. This can be helpful for C++ template metaprogramming. For example:

    template<class ViewType>
    void callBlas (const ViewType& A) {
      typedef typename ViewType::array_layout array_layout;
      if (std::is_same<array_layout, LayoutLeft>::value) {
        callSomeBlasFunction (A.data(), ...);
      } else {
        throw std::invalid_argument ("A is not LayoutLeft");
      }
    }

