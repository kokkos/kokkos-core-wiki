# 15. SIMD Types

## Background

Through the history of high-performance computing using CPUs, there has been a struggle to get software to effectively make use of CPU vector registers and instructions through compiler auto-vectorization.
An interesting look into this struggle can be found in [this blog](https://pharr.org/matt/blog/2018/04/18/ispc-origins), which gives some insight into how Intel compiler auto-vectorization has drawbacks compared to programming models such a CUDA for NVIDIA GPUs.
A key quote, attributed to Stanford's Tim Foley, states that "auto-vectorization is not a programming model".
This is essentially the heart of the issue: auto-vectorization is not a two-way conversation between programmer and machine. It is more like a black box that almost always fails to work.
The [dissertation](https://www.researchgate.net/profile/Matthias_Kretz2/publication/295253746_Extending_C_for_Explicit_Data-Parallel_Programming_via_SIMD_Vector_Types/links/56c8c99208ae5488f0d6ffa0.pdf) of Matthias Kretz, one of the key driving forces behind SIMD types being proposed to the ISO standard of the C++ language, describes similar motivations.
Since Kokkos is a programming model that strives to deliver high performance on vector-capable CPUs, it makes sense for it to provide these SIMD types due to the limitations of alternative approaches.

## The Basic Idea

The idea behind SIMD types is to step just one layer of abstraction above hand-coding Intel intrinsic calls.
The design of SIMD types recognizes that the vector intrinsics provided by many CPU vendors are fairly similar and that users would not like to hand-code for one specific vendor at a time.
As such, the SIMD types are a C++ representation of vector registers and their methods explicitly call vendor-specific vector intrinsic instructions.
The vendor-specific portions are abstracted away from user code through template parameters.

The reason that SIMD types are so effective at producing high-performance code is that users are directly expressing what the vector parallelism is that they would like, and doing so in a way that is guaranteed to generate the vector instructions they expect.
When using SIMD types, the compiler cannot fail to auto-vectorize, because auto-vectorization is not part of the picture. 
Users are more clearly able to reason about the available parallelism in their code in ways that an auto-vectorizer is almost never able to do.
By allowing users to have direct control of how vectorization is done while protecting them from both vendor lock-in and the odd quirks of different vendor instruction sets, SIMD types allow users to write performance portable code accross CPUs and also GPUs.

## An Example

Suppose that we have the following loop that we would like to take advantage of CPU vectorization:

```c++
double* x;
double* y;
double* z;
double* r;
int n;
for (int i = 0; i < n; ++i) {
  r[i] = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
}
```

Here is one way to convert such a loop to use Kokkos SIMD types:

```c++
#include <Kokkos_SIMD.hpp>

using simd_type = Kokkos::Experimental::native_simd<double>;
using tag_type = Kokkos::Experimental::element_aligned_tag;
constexpr int width = int(simd_type::size());
assert(n % width == 0);
for (int i = 0; i < n; i += width) {
  simd_type sx(x + i, tag_type());
  simd_type sy(y + i, tag_type());
  simd_type sz(z + i, tag_type());
  simd_type sr = Kokkos::sqrt(sx * sx + sy * sy + sz * sz);
  sr.copy_to(r + i, tag_type());
}
```

`Kokkos::Experimental::native_simd<double>` is the basic SIMD type for a vector register
containing values of 64-bit floating-point type.
Constructing one given a pointer (and alignment tag) will execute a vectorized load instruction,
and `copy_to` will generate a vectorized store instruction.
Often these instructions are the only way to get the full memory bandwidth from your CPU.
The SIMD type provides the basic math operators, so the three multiplications
and two additions are guaranteed to turn into vector instructions for multiplication and addition.
The `Kokkos::sqrt` overload will call a vector instruction for computing the square roots
of values if the CPU supports it.

In other words, we now have a C++ code which has no vendor-specific stuff in it but is guaranteed
to emit exactly the right vendor vector instructions depending on which CPU type Kokkos
is compiled for (`KOKKOS_ARCH` configurations).
It is also still fairly readable, especially the portion that deals with the mathematical operations.

If a CPU supports 256-bit vector registers, then this code will process 4 `double`s at a time (width = 4), and get close to 4X speedup depending on the situation.

## Dealing with the Remainder

In the above example, we skipped over a troublesome pitfall by asserting that the size of the data `n` that we are operating on is evenly divisible by the vector width.
There are at least three major approaches to dealing with this issue in general:

1. Enforce that the data size is always a multiple of the vector width.
   One way to do this is by padding, which means allocate more than you need until it is a multiple and ignore the extra values.
   Another way to do this is by storing types which are also the same size, which you can do by using the same SIMD types:

   ```c++
   #include <Kokkos_SIMD.hpp>
   
   using simd_type = Kokkos::Experimental::native_simd<double>;
   simd_type* x;
   simd_type* y;
   simd_type* z;
   simd_type* r;
   int n;
   for (int i = 0; i < n; ++i) {
     r[i] = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
   }
   ```
   
   Kokkos is working on adding special storage types to further support this use case and allow interoperability with `Kokkos::View`.

   Note that this approach looks cleaner in the example but can have wide-reaching consequences because the data type of the original data has changed, and may need to change in other places throughout a large code base.

2. Handle the remainder differently. For example, use a loop of non-vectorized code to compute the remainder values:

   ```c++
   #include <Kokkos_SIMD.hpp>
 
   using simd_type = Kokkos::Experimental::native_simd<double>;
   using tag_type = Kokkos::Experimental::element_aligned_tag;
   constexpr int width = int(simd_type::size());
   int i;
   for (i = 0; i + width <= n; i += width) {
     simd_type sx(x + i, tag_type());
     simd_type sy(y + i, tag_type());
     simd_type sz(z + i, tag_type());
     simd_type sr = Kokkos::sqrt(sx * sx + sy * sy + sz * sz);
     sr.copy_to(r + i, tag_type());
   }
   for (; i < n; ++i) {
     r[i] = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
   }
   ```
  
   The main drawback of this approach is the code duplication / repetition.

3. Use masks in each iteration:

   ```c++
   #include <Kokkos_SIMD.hpp>
 
   using simd_type = Kokkos::Experimental::native_simd<double>;
   using mask_type = Kokkos::Experimental::native_simd_mask<double>;
   using tag_type = Kokkos::Experimental::element_aligned_tag;
   constexpr int width = int(simd_type::size());
   for (int i = 0; i < n; i += width) {
     mask_type mask([] (std::size_t lane) { return i + int(lane) < n; });
     simd_type sx;
     simd_type sy;
     simd_type sz;
     where(mask, sx).copy_from(x + i, tag_type());
     where(mask, sy).copy_from(y + i, tag_type());
     where(mask, sz).copy_from(z + i, tag_type());
     simd_type sr = Kokkos::sqrt(sx * sx + sy * sy + sz * sz);
     where(mask, sr).copy_to(r + i, tag_type());
   }
   ```
  
   The main drawback of this approach is the slight overhead of using masked loads and stores,
   but it nicely handles data from sources that have neither padded nor changed their data type
   without any code repetition.

## Vectorization of Library Code

By using templating, complex mathematical library code can automatically take full advantage of vectorization without changing:

```c++
template <class T>
KOKKOS_FUNCTION void quadratic_formula(
    T const& a,
    T const& b,
    T const& c,
    T& x1,
    T& x2)
{
  T discriminant = b * b - 4 * a * c;
  T sqrt_discriminant = Kokkos::sqrt(discriminant);
  x1 = (-b + sqrt_discriminant) / (2 * a);
  x2 = (-b - sqrt_discriminant) / (2 * a);
}
```

When instantiated with `T=double`, this function behaves in the classic, familiar, serial sense.
If we simply instantiate it with `T=Kokkos::Experimental::native_simd<double>`, it still compiles just the same but now every mathematical operation is guaranteed to emit a vector instruction and the function can compute 4 quadratic formulas at a time on a 256-bit vector CPU.

Note that Kokkos takes special care to ensure everything that can be done with `double` can also be done with SIMD types, including the multiplication by integer literals `4` and `2` in this example code.

## Conditionals

### Conditional assignment

One of the more difficult things to do with SIMD types is conditional logic. Consider the following code which is responsible for ensuring that the value `x` is not negative (we will ignore the existence of `max` functions for this discussion because it makes for a nice simple example):

```c++
double x;
if (x < 0) x = 0;
```

We cannot naively use SIMD types in this scenario, because `x < 0` is not a boolean value, instead it is a `simnd_mask<double, Abi>` object which represents possibly multiple booleans.

```c++
Kokkos::Experimental::native_simd<double> x;
if (x < 0 /* <- this is not a boolean */) x = 0;
```

The ISO C++ consistent solution is to use `where` expressions as follows:

```c++
Kokkos::Experimental::native_simd<double> x;
where(x < 0, x) = 0;
```

### Ternary operator

At the time of this writing, there is also a common practice of using a function that mimics the behavior of the ternary conditional operator `a ? b : c` in a SIMD sense. This means the following are functionally equivalent:

```c++
bool a;
double b;
double c;
auto d = a ? b : c;
```

```c++
Kokkos::Experimental::native_simd_mask<double> a;
Kokkos::Experimental::native_simd<double> b;
Kokkos::Experimental::native_simd<double> c;
auto d = Kokkos::Experimental::condition(a, b, c);
```

```c++
Kokkos::Experimental::native_simd_mask<double> a;
Kokkos::Experimental::native_simd<double> b;
Kokkos::Experimental::native_simd<double> c;
auto d = c;
where(a, d) = b;
```

The roadmap regarding the ternary operator is as follows: ISO C++ will likely add the ability to overload this operator in later versions of the language, and the standard library's SIMD types will overload it.

It is recommended that programmers use `where` expressions as much as possible for conditional logic when using Kokkos SIMD types, because this is consistent with the library solution currently proposed to ISO C++ without relying on non-standard functions or language features not yet available.

### Reductions for performance

One frustrating aspect of the solutions for conditional logic above is that computations are not skipped, they are simply masked out. Consider the following serial logic:

```c++
bool a;
double b = 1.0;
if (a) b = very_expensive_function(c, d, e);
```

When using `if` statements, `very_expensive_function` is not executed at all unless `a` is `true`. However, in SIMD mode:

```c++
Kokkos::Experimental::native_simd_mask<double> a;
Kokkos::Experimental::native_simd<double> b = 1.0;
where(a, b) = very_expensive_function(c, d, e);
```

Now `very_expensive_function` is always being executed. What if the probability of `a` being `true` is very low? We would want to skip the computation of `very_expensive_function` if at all possible.

For this, we have boolean reductions across masks called `all_of`, `none_of`, and `any_of`:

```c++
Kokkos::Experimental::native_simd_mask<double> a;
Kokkos::Experimental::native_simd<double> b = 1.0;
if (Kokkos::Experimental::any_of(a)) {
  where(a, b) = very_expensive_function(c, d, e);
}
```

Now `very_expensive_function` will only be executed if any of the boolean values in the mask `a` are `true`. If it is common that all of the boolean values in `a` are `false`, then we will spend a lot less time in `very_expensive_function`.
