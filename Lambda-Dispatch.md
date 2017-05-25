# Lambda Dispatch

## What is a lambda ?

In the context of the C++11 language standard, a lambda is a new syntactic feature
that essentially automatically creates a functor.
Consider the standard library's templated `std::find_if` function.
Prior to C++11, functors were the way to use this function:

```cpp
struct IsTheSameAsX {
  int x;
  IsTheSameAsX(int given_x) { x = given_x; }
  bool operator()(int const& y) const { return y == x; }
};

std::vector<int>::iterator find_equal(std::vector<int> const& v, int x) {
  return std::find_if(v.begin(), v.end(), IsTheSameAsX(x));
}
```

In C++11, the lambda syntax
```
[captures] (arguments) -> result_type { code }
```
allows automatically creating functors without needing to give them a name,
and most importantly it automatically handles the member variables of the functor.
For example, the following should be exactly equivalent to our previous usage
of `std::find_if`:

```cpp
std::vector<int>::iterator find_equal(std::vector<int> const& v, int x) {
  return std::find_if(v.begin(), v.end(), [=] (int const& y) { return y == x; });
}
```

The `[=]` capture list means two things:
 - The compiler should determine which variables are to be captured, based on the code inside the lambda 
   (alternatively one may explicitly list the variables to capture).
 - All such captures should be done by value (indicated by the `=`), as opposed to by reference
   (which would be indicated by an `&`).

One can see in comparing the examples above that lambdas offer a great advantage
in terms of the number of lines of code.
Also, because the functor created is an unnamed implicit type, there is no need for
developers to come up with names like `IsTheSameAsX` or place its definition outside
the function.
Finally, note that the result type of this lambda was automatically determined based
on the `return` statement inside of it.

### Lambdas inside classes

As it will be relevant to the discussion of lambdas in Kokkos, it is important
to discuss lambda capture behavior as it applies to member variables of a class.
Consider this code:

```cpp
class LotsToDo {
 public:
  size_t which_is_it() {
    auto it = std::find_if(v.begin(), v.end(), [=] (int const& y) { return y == x; });
    return it - v.begin();
  }
 private:
  std::vector<int> v;
  int x;
};
```

The key aspect we'll focus on is the fact that the variable `x` is a member of
class `LotsToDo`, which will make its capture rather non-intuitive.

Conceptually, C++ compilers replace every use of a class's member variable
`x` with `this->x` at some point during compilation.
Compilers also conceptually add an implicit `this` argument to each member
function.
The key thing to note is that (conceptually) these transformations happen *before*
the considerations associated with variable capture in a lambda.
This means the intermediate conceptual state looks like:

```cpp
struct LotsToDo {
  std::vector<int> v;
  int x;
};

size_t LotsToDo::which_is_it(LotsToDo* this) {
  auto it = std::find_if(this->v.begin(), this->v.end(), [=] (int const& y) { return y == this->x; });
  return it - this->v.begin();
}
```

Now the situation looks quite different to the compiler.
`x` is no longer a variable in the `which_is_it` function, rather it is accessed
indirectly via `this`.
So the compiler decides to capture the `this` pointer, meaning the intermediate
conceptual state after lambdas are converter to functors looks somewhat like this:

```cpp
struct LotsToDo {
  std::vector<int> v;
  int x;
};

struct Lambda {
  LotsToDo* this;
  Lambda(LotsToDo* given_this) { this = given_this; }
  bool operator()(int const& y) { return y == this->x; }
};

size_t LotsToDo::which_is_it(LotsToDo* this) {
  auto it = std::find_if(this->v.begin(), this->v.end(), Lambda(this));
  return it - this->v.begin();
}
```

Under normal circumstances, this actually works fine despite the fact
that it works rather differently than developers initially expect.
We'll see later that it causes problems with Kokkos + CUDA.

Note that if we wanted things to work more intuitively, one way to do that
would be to create a local variable copy of the member variable:

```cpp
class LotsToDo {
 public:
  size_t which_is_it() {
    int local_x = this->x;
    auto it = std::find_if(v.begin(), v.end(), [=] (int const& y) { return y == local_x; });
    return it - v.begin();
  }
 private:
  std::vector<int> v;
  int x;
};
```

`local_x` will not undergo any transformations of indirection, so it will
be captured as expected:

```cpp
struct LotsToDo {
  std::vector<int> v;
  int x;
};

struct Lambda {
  int local_x;
  Lambda(int given_local_x) { local_x = given_local_x; }
  bool operator()(int const& y) { return y == local_x; }
};

size_t LotsToDo::which_is_it(LotsToDo* this) {
  int local_x = this->x;
  auto it = std::find_if(this->v.begin(), this->v.end(), Lambda(this));
  return it - this->v.begin();
}
```

In addition, the C++17 standard includes a special capture list for lambdas (`[=,*this]`) which
will force the lambda to capture the object instead of the pointer
(the Kokkos team was involved in this addition to the C++ standard).
If we use that feature in our example:

```cpp
class LotsToDo {
 public:
  size_t which_is_it() {
    auto it = std::find_if(v.begin(), v.end(), [=,*this] (int const& y) { return y == x; });
    return it - v.begin();
  }
 private:
  std::vector<int> v;
  int x;
};
```

The conceptual generated code is:

```cpp
struct LotsToDo {
  std::vector<int> v;
  int x;
};

struct Lambda {
  LotsToDo star_this;
  Lambda(LotsToDo const& given_star_this) { star_this = given_star_this; }
  bool operator()(int const& y) { return y == star_this.x; }
};

size_t LotsToDo::which_is_it(LotsToDo* this) {
  auto it = std::find_if(this->v.begin(), this->v.end(), Lambda(*this));
  return it - this->v.begin();
}
```

Note that this is making a deep copy of a `std::vector<int>`, so when using this
feature one should review the contents of the class being captured to avoid
unwanted consequences of capturing unneeded variables.

## Lambdas in Kokkos

Like `std::find_if`, `Kokkos::parallel_for` accepts user-defined functors to essentially
implement a zero-overhead callback to user code.
If used properly, lambdas can provide their intended benefits to Kokkos-calling code as well.
However, there are additional restrictions imposed by CUDA and to a lesser extent OpenMP,
which are the Kokkos backends.

### CUDA support for lambdas

CUDA is currently Kokkos' only backend which can utilize GPUs, and it makes use of a custom
C++ compiler called NVCC.
This compiler has its own set of supported and non-supported C++ features, separate from
the usual host (CPU) compiler.
At the time of this writing, the two most widely used versions of CUDA are version 7.5 and version 8.0.

CUDA defines function attributes that define how a function will be compiled by NVCC:

1. `__host__` A version of this function will be compiled for execution on the CPU host.
1. `__device__` A version of this function will be compiled for execution on the GPU device.
1. `__global__` This function is a kernel dispatched from the host.
   Users of Kokkos need not know about `__global__`, only internal Kokkos functions have this attribute.

As an example, this function can be called at any point in a code
that is compiled by NVCC:

```cpp
__host__ __device__ double funny_multiply(double a, double b) {
  return a * b - 1e-6;
}
```

Kokkos provides convenience macros which expand to these attributes when using CUDA and
expand to nothing otherwise:

```cpp
#ifdef KOKKOS_HAVE_CUDA
#define KOKKOS_INLINE_FUNCTION __host__ __device__
#else
#define KOKKOS_INLINE_FUNCTION
#endif
```

These macros exist to satisfy the "single-source" principle
that Kokkos strives for, i.e. the following code using Kokkos
can be compiled for any machine by changing only configuration settings:

```cpp
KOKKOS_INLINE_FUNCTION double funny_multiply(double a, double b) {
  return a * b - 1e-6;
}
```

Note that by default the attributes are both `__host__` and `__device__`, which instructs
the NVCC compiler to generate both CPU and GPU versions of the annotated function.

Like functions, lambdas can have CUDA attributes.
CUDA 7.5 and CUDA 8.0 both support the `__device__` attribute on a lambda (`[=] __device__ (int x) { ... }`).
These are referred to as "device lambdas".
CUDA 8.0 supports lambdas with both attributes (`[=] __host__ __device__ (int x) { ... }`),
which are referred to as "host-device lambdas".

Note the trade-off with CUDA 7.5: one can use lambdas, but they will only be executable
on the GPU, so one cannot compile a single "kernel" which can execute on both the CPU and GPU.

CUDA versions prior to 7.5 do not support device lambdas !

Ideally, code like the following should work with both versions of CUDA:

```cpp
int main() {
  int array_size = 10;
  Kokkos::View<double*> array("array", array_size);
  double factor = 42.0;
  Kokkos::parallel_for(array_size, [=] __device__ (int i) {
    array(i) = factor * i;
  });
}
```

Note that the variable `array`, which is a `Kokkos::View`, is being captured by value.
The conceptual expanded code is:

```cpp
struct Lambda {
  Kokkos::View<double*> array;
  double factor;
  Lambda(Kokkos::View<double*> const& given_array, double const& given_factor) {
    array = given_array;
    factor = given_factor;
  }
  __device__ void operator()(int i) {
    array(i) = factor * i;
  }
};

int main() {
  int array_size = 10;
  Kokkos::View<double*> array("array", array_size);
  double factor = 42.0;
  Kokkos::parallel_for(array_size, Lambda(array, factor));
}
```

As the number of variables captured by a lambda increases, so does its advantage
over a functor in terms of lines of code.

Now we can present the set of `LAMBDA` macros offered by Kokkos.
Kokkos still considers CUDA lambdas
a somewhat experimental feature, so they are disabled by default.
If building Kokkos as part of Trilinos, CUDA lambdas can be
enabled with this CMake configuration flag:

```
-DKokkos_ENABLE_Cuda_Lambda:BOOL=ON
```

`KOKKOS_LAMBDA` will be defined to `[=] __device__` or `[=] __host__ __device__`,
depending on your CUDA version.
Without CUDA it is simply `[=]`.

If C++17 and CUDA 8.0 are used, `KOKKOS_CLASS_LAMBDA` will be defined to
`[=,*this] __host__ __device__`.
If C++17 is used without CUDA 8.0, `KOKKOS_CLASS_LAMBDA` is just `[=,*this]`.
Without C++17, `KOKKOS_CLASS_LAMBDA` is not defined.
Note that there is no variant with only `__device__` (as there was for `KOKKOS_LAMBDA` for CUDA 7.5),
because `[=,*this]` is only supported in CUDA 8.0, so without CUDA 8.0 `KOKKOS_CLASS_LAMBDA` does not work on GPUs,
even with CUDA 7.5.

Thus, our CUDA-specific Kokkos example above should look like this in typical
usage of Kokkos:

```cpp
int main() {
  int array_size = 10;
  Kokkos::View<double*> array("array", array_size);
  double factor = 42.0;
  Kokkos::parallel_for(array_size, KOKKOS_LAMBDA(int i) {
    array(i) = factor * i;
  });
}
```

### Lambdas inside classes with CUDA

Recall from the section above on lambdas in classes how the capture of class member variables
is actually done by capturing the `this` pointer.
Lets return to our prior CUDA-specific example and put it inside a class to demonstrate the issues.
If we use `[=] __device__` as our lambda, like so:

```cpp
class Fancy {
 public:
  void set_values() {
    Kokkos::parallel_for(array_size, KOKKOS_LAMBDA (int i) {
      array(i) = factor * i;
    });
  }
 private:
  int array_size;
  Kokkos::View<double*> array;
  double factor;
}
```

We get the `this`-capturing behavior:

```cpp
struct Fancy {
  int array_size;
  Kokkos::View<double*> array;
  double factor;
};

struct Lambda {
  Fancy* this;
  Lambda(Fancy* given_this) { this = given_this; }
  __device__ void operator()(int i) const {
    this->array(i) = this->factor * i;
  }
};

void Fancy::set_values(Fancy* this) {
  Kokkos::parallel_for(this->array_size, Lambda(this));
}
```

Notice that although this will compile, it will crash at runtime
because `Lambda::this` is a pointer to an object in CPU memory
(we are assuming that the `Fancy` object is in CPU memory).
The GPU will attempt to access `this->array` and `this->factor`,
which is an illegal memory access from the GPU to CPU memory.

Now see how using `KOKKOS_CLASS_LAMBDA` can help:

```cpp
class Fancy {
 public:
  void set_values() {
    Kokkos::parallel_for(array_size, KOKKOS_CLASS_LAMBDA (int i) {
      array(i) = factor * i;
    });
  }
 private:
  int array_size;
  Kokkos::View<double*> array;
  double factor;
}
```

Is transformed into this:

```cpp
struct Fancy {
  int array_size;
  Kokkos::View<double*> array;
  double factor;
};

struct Lambda {
  Fancy star_this;
  Lambda(Fancy const& given_star_this) { star_this = given_star_this; }
  __device__ void operator()(int i) const {
    star_this.array(i) = star_this.factor * i;
  }
};

void Fancy::set_values(Fancy* this) {
  Kokkos::parallel_for(this->array_size, Lambda(*this));
}
```

As long as each member variable of the `Fancy` class
are okay to copy by value onto the GPU, this will work properly.
Simple types `int` and `double` are always okay,
and `Kokkos::View` is specially designed to be okay.
However, there would be an issue if class `Fancy`
also had a member which was of type `std::vector<int>`,
for example, because this type has constructors and
a destructor which cannot be called from the GPU.

Another known limitation is that a CUDA lambda may
not be used inside a class member function that is
private or protected (TODO: give an explanation of this).
