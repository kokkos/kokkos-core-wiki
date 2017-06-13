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

The string argument is a label, which Kokkos uses for debugging. Different Views may have the same label. The ellipses indicate some integer dimensions specified at run time. Users may also set some dimensions at compile time. For example, the following View has two dimensions. The first one (represented by the asterisk) is a run-time dimension, and the second (represented by [3]) is a compile-time dimension. Thus, the View is an N by 3 array of type double, where N
is specified at run time in the View's constructor.

    const size_t N = ...;
    View<double*[3]> b ("another label", N);

Views may have up to (at least) 8 dimensions, and any number of these may be run-time or compile-time. The only limitation is that the run-time dimensions (if any) must appear first, followed by all the compile-time dimensions (if any). For example, the following are valid three-dimensional View types:

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

Note that the above used constructor is not necessarily available for all view types. Specific Layouts or Memory Spaces may require more specialized allocators. This will be discussed later.

Another important thing to keep in mind is that a `View` handle is a stateful object. It is not legal to create a `View` handle from raw memory by typecasting a pointer. To call any operator on a `View` its constructor must have been called before; this includes the assignment operator. If it is necessary to initialize raw memory with a `View` handle, one can legally do so using a move constructor ("placement new"). The above has nothing to do with the data a `View` is referencing. It is completely legal to give a typecast pointer to the constructor of an unmanaged `View`.

    View<int*> * a_ptr = (View<int*>*) malloc(10*sizeof(View<int*);
    a_ptr[0] = View<int*>("A0",1000); // This is illegal
    new(&a_ptr[1]) View<int*>("A1",10000); // This is legal 


### 6.2.2 What types of data may a View contain?

C++ lets users construct data types that may "look like" numbers in terms of syntax but do arbitrarily complicated things inside. Some of those things may not be thread safe, like unprotected updates to global state. Others may perform badly in parallel, like fine-grained dynamic memory allocation. Therefore, it is strongly advised to use only simple data types inside Kokkos Views. Users may always construct a View whose entries are

* built-in data types (``plain old data''), like `int` or `double`, or
* structs of built-in data types.

While it is in principal possible to have Kokkos Views of arbitrary objects, there are a number of restrictions. The `T` type must have a default constructor and destructor. For portability reasons `T` is not allowed to have virtual functions and one is not allowed to call functions with dynamic memory allocations inside of kernel code. Furthermore, assignment operators as well as default constructor and deconstructor must be marked with `KOKKOS_[INLINE_]FUNCTION`. Keep in mind that the semantics of the resulting View are a combination of the Views 'view' semantics and the behaviour of the element type.

### 6.2.3 Const Views

A view can have const data semantics (i.e. its entries are read-only) by specifying a `const` data type.  It is a compile-time error to assign to an entry of a "const View". Assignment semantics are equivalent to a pointer to const data. A const View means the _entries_ are const; you may still assign to a const View. `View<const double*> corresponds exactly to `const double*`, and `const View<double*>` to `double* const`. Therefore, it does not make sense to allocate a const View, since you could not obtain a non-const view of the same data and you can not assign to it. You can however assign a non-const view to a const view. Here is an example:

    const size_t N0 = ...;
    View<double*> a_nonconst ("a_nonconst", N0);

    // Assign a nonconst View to a const View
    View<const double*> a_const = a_nonconst;
    // Pass the const View to some read-only function.
    const double result = readOnlyFunction (a_const);


Const Views often enables the compiler to optimize more aggressively by allowing it to reason about possible write conflicts and data aliasing. For example, in a vector update `a(i+1)+=b(i)` with skewed indexing, it is safe to vectorize if `b` is a View of const data.
