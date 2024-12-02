# Dealing with function pointers - when you really have to

Function pointers are something to be avoided when writing Kokkos code (or any portable code for that matter).
However, if you really have to it can be made to work - with some effort.
Here we will give some explanation of why it is complicated, and how to work around the hurdles.

## The naive approach and why it doesn't work.

Let's start with some simple thing, where `SomeClass` contains a function pointer which you want to use inside a `KOKKOS_FUNCTION` marked function:

```c++
struct SomeClass {
  void (*bar)();
  KOKKOS_FUNCTION void print() const {
    bar();
  }
};
```

Going forward we will use a simple examplar function:

```c++
KOKKOS_INLINE_FUNCTION void foo() {
  KOKKOS_IF_ON_HOST(printf("foo called from host\n");)
  KOKKOS_IF_ON_DEVICE(printf("foo called from device\n");)
}
```

This function leverages the `KOKKOS_IF_ON_HOST` and `KOKKOS_IF_ON_DEVICE` macros so we can tell which version we got.

Putting it all together into a fully self contained source and lets try to call it both on host and on device:

```c++
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION void foo() {
  KOKKOS_IF_ON_HOST(printf("foo called from host\n");)
  KOKKOS_IF_ON_DEVICE(printf("foo called from device\n");)
}

struct SomeClass {
  void (*bar)();
  KOKKOS_FUNCTION void print() const {
    bar();
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    SomeClass A;
    A.bar = &foo;
    // Call it plain on host
    A.print();

    // Call it inside a host parallel for
    printf("I can use the function pointer in a host parallel_for!\n");
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A.print();
    });
    Kokkos::fence();
    printf("Now I will crash if we compiled for CUDA/HIP\n");
    // Try to call it on device
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A.print();
    });
    Kokkos::fence();
    printf("Never got here in CUDA/HIP\n");

  }
  Kokkos::finalize();
}
```

This worked on host (both inside and outside a `parallel_for` but crashes inside a device kernel.
Here is the output:

```
foo called from host
I can use the function pointer in a host parallel_for!
foo called from host
Now I will crash if we compiled for CUDA/HIP
cudaDeviceSynchronize() error( cudaErrorIllegalAddress): an illegal memory access was encountered /home/crtrott/Kokkos/kokkos/core/src/Cuda/Kokkos_Cuda_Instance.cpp:153
Backtrace:
[0x4422e3] 
[0x4249d5] 
[0x43085d] 
[0x430a5e] 
[0x429855] 
[0x40843e] 
[0x7f7fca5ba7e5] __libc_start_main
[0x409cce] 
Aborted (core dumped)
```

*The function pointer we created was for a host function: you can not use it on device same as you can't dereference data pointers to host data!*

## Getting a device function pointer

We actually can get the pointer to the device version of our function `foo`. But we can only do that *on the device*!
That means we need to run a little Kokkos kernel get the function pointer there, and copy it somehow back to the host so we can set the host object that way.

To do this we need a device `View` of a `SomeClass` instance, set the function pointer in the device code, and `deep_copy` it back into our host instance.

```c++
    SomeClass A;
    Kokkos::View<SomeClass> A_v("A_v");
    // Now init the function pointer on device:
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A_v().bar = &foo;
    });
    // copy the device instance back to the host
    Kokkos::deep_copy(A, A_v);
```

We are leveraging here the fact that `deep_copy` allows you to copy from and to a host scalar value, from a `View` of rank-0.

If we do this `A` will contain a function pointer to a device function. That means we can capture `A` into a parallel region and execute it on the device, but now it will crash on the host.

```c++
#include <Kokkos_Core.hpp>
#include <cmath>

KOKKOS_INLINE_FUNCTION void foo() {
  KOKKOS_IF_ON_HOST(printf("foo called from host\n");)
  KOKKOS_IF_ON_DEVICE(printf("foo called from device\n");)
}

struct SomeClass {
  void (*bar)();
  KOKKOS_FUNCTION void print() const {
    bar();
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    SomeClass A;
    Kokkos::View<SomeClass> A_v("A_v");
    // Now init the function pointer on device:
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A_v().bar = &foo;
    });
    // copy the device instance back to the host
    Kokkos::deep_copy(A, A_v);

    printf("Now I can capture A in a device kernel and use it there!\n");
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A.print();
    });
    Kokkos::fence();
    printf("But now I will crash on the host :-(\n");
    A.print();
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A.print();
    });
    Kokkos::fence();
    printf("Never got here in CUDA/HIP builds\n");
  }
  Kokkos::finalize();
}
```

If you run this with a CUDA build you get this output:

```
Now I can capture A in a device kernel and use it there!
foo called from device
But now I will crash on the host :-(
Segmentation fault (core dumped)
```

## Creating a dual function pointer object

To do better than this we need something which contains both the device function pointer and the host function pointer.
And it should call the right thing depending on where you call it from.
We can in-fact write such a class:

```c++
template<class FPtr>
struct DualFunctionPtr {
  FPtr h;
  FPtr d;
  template<class ... Args>
  KOKKOS_FUNCTION
  auto operator()(Args...args) const {
    KOKKOS_IF_ON_HOST( return h(args...); )
    KOKKOS_IF_ON_DEVICE( return d(args...); )
  }
};
```

This class is templated on the function pointer type, contains a pointer for both the host and the device version, and has a templated operator that forwards all the arguments, and calls the appropriate function pointer depending on call site.

We can use this class similar to `std::function` inside `SomeClass`:

```c++
struct SomeClass {
  DualFunctionPtr<decltype(&foo)> bar;
  KOKKOS_FUNCTION void print() const {
    bar();
  }
};
```

However we still need to initialize the two function pointers, using the device initialization approach from before, and also initializing the host side. Note: the order of initialization and `deep_copy` matters, because `deep_copy` will overwrite both member poitners in our wrapper class:

```c++
    SomeClass A;
    Kokkos::View<SomeClass> A_v("A_v");
    // Now init the function pointer on device:
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A_v().bar.d = &foo;
    });
    // copy the device instance back to the host
    Kokkos::deep_copy(A, A_v);
    // Now init the host ptr
    A.bar.h = &foo;
```

With that we have a fully working code where we can use our instance `A` of `SomeClass` on the host and on the device:

```c++
#include <Kokkos_Core.hpp>

KOKKOS_INLINE_FUNCTION void foo() {
  KOKKOS_IF_ON_HOST(printf("foo called from host\n");)
  KOKKOS_IF_ON_DEVICE(printf("foo called from device\n");)
}

template<class FPtr>
struct DualFunctionPtr {
  FPtr h;
  FPtr d;
  template<class ... Args>
  KOKKOS_FUNCTION
  auto operator()(Args...args) const {
    KOKKOS_IF_ON_HOST( return h(args...); )
    KOKKOS_IF_ON_DEVICE( return d(args...); )
  }
};

struct SomeClass {
  DualFunctionPtr<decltype(&foo)> bar;
  KOKKOS_FUNCTION void print() const {
    bar();
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    SomeClass A;
    Kokkos::View<SomeClass> A_v("A_v");
    // Now init the function pointer on device:
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A_v().bar.d = &foo;
    });
    // copy the device instance back to the host
    Kokkos::deep_copy(A, A_v);
    // Now init the host ptr
    A.bar.h = &foo;

    printf("Now I can capture A in a device kernel and use it there!\n");
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A.print();
    });
    Kokkos::fence();
    printf("And I can capture A on the host and use it there\n");
    A.print();
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,1), KOKKOS_LAMBDA(int) {
      A.print();
    });
    Kokkos::fence();
    printf("I didn't crash whoohoo!!\n");
  }
  Kokkos::finalize();
}
```

While this generally works (and note this code will also work if you simply compile for host-only) there may be architectures in the future where this fails.
Also we in principle strongly discourage the use of function pointers - the amount of possible problems you run into is fairly significant.
Furthermore, the visibility of the functions is important. In the example above all functions were in the same translation unit.
You may need to use *relocatable device code* if that is not the case.

