# Kokkos Detection Idiom

The Kokkos Detection Idiom is used to recognize, in a SFINAE-friendly way, the validity of any C++ expression.

Header File: `Kokkos_DetectionIdiom.hpp`

The Kokkos Detection Idiom is based upon the detection idiom from Version 2 of the C++ Extensions for
Library Fundamentals, ISO/IEC TS 19568:2017, a draft of which can be found at
<https://cplusplus.github.io/fundamentals-ts/v2.html#meta.detect>.
The original C++ proposal can be found at <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4436.pdf>.
Those are excellent documents to learn more about the design details and how the mechanism works.

The API:


```
// VOID_T and DETECTOR are exposition-only and not intended to be used directly.

// Convienent metafunction to leverage SFINAE
template<class...>
using VOID_T = void;

// Primary template for types not supporting the archetypal Op<Args...>
template<class Default, class /* AlwaysVoid */, template<class...> class /* Op */, class... /* Args */>
struct DETECTOR {
    using value_t = std::false_type;
    using type    = Default;
};

// Specialization for types supporting the archtypal Op<Args...>
template<class Default, template<class...> class Op, class... Args>
struct DETECTOR<Default, VOID_T<Op<Args...>>, Op, Args...> {
    using value_t = std::true_type;
    using type    = Op<Args...>;
};

namespace Kokkos {

// Simplifiation of the type returned by detected_t for types not supporting the archtype provided
struct nonesuch {
    nonesuch(nonesuch&&) = delete;
    ~nonesuch() = delete;
};

// is_detected is an alias for std::true_type if Op<Args...> is a valid type
//  otherwise, an alias for std::false_type

template <template <class...> class Op, class... Args>
using is_detected =
    typename DETECTOR<nonesuch, void, Op, Args...>::value_t;

// detected_t is an alias for Op<Args...> if Op<Args...> is a valid type
//  otherwise, an alias for Kokkos::nonesuch

template <template <class...> class Op, class... Args>
using detected_t = typename DETECTOR<nonesuch, void, Op, Args...>::type;

// detected_or_t is an alias for Op<Args...> if Op<Args...> is a valid type
//  otherwise, an alias for Default

template <class Default, template <class...> class Op, class... Args>
using detected_or_t = typename DETECTOR<Default, void, Op, Args...>::type;

// is_detected_exact is an alias for std::true_type if Op<Args...> is the same type as Expected
//  otherwise, an alias for std::false_type

template <class Expected, template <class...> class Op, class... Args>
using is_detected_exact = std::is_same<Expected, detected_t<Op, Args...>>;

// is_detected_convertible is an alias for std::true_type if Op<Args...> is convertible to To
//  otherwise, an alias for std::false_type

template <class To, template <class...> class Op, class... Args>
using is_detected_convertible =
    std::is_convertible<detected_t<Op, Args...>, To>;

// C++17 or later convenience variables

template <template <class...> class Op, class... Args>
inline constexpr bool is_detected_v = is_detected<Op, Args...>::value;

template <class Expected, template <class...> class Op, class... Args>
inline constexpr bool is_detected_exact_v =
    is_detected_exact<Expected, Op, Args...>::value;

template <class Expected, template <class...> class Op, class... Args>
inline constexpr bool is_detected_convertible_v =

} // Kokkos namespace
```

Usage examples:

Suppose we needed to write a type trait to detect if a given type T
is copy assignable.  First we write an archtype helper alias:

```
template<class T>
using copy_assign_t = decltype(std::declval<T&>() = std::declval<T const&>());
```

Then the trait can be easily expressed as:

```
template<class T>
using is_copy_assignable = Kokkos::is_detected<copy_assign_t, T>;
```

If we also wanted to check that the return type of the copy assignment
was T&, we would use:

```
template<class T>
using is_canonical_copy_assignable = Kokkos::is_detected_exact<T&, copy_assign_t, T>;
```

Suppose we want to use a nested `MyType::difference_type` if it
exists, otherwise we want to use `std::ptrdiff_t`:

First we write an archtype helper alias:

```
template<class T>
using diff_t = typename T::difference_type;
```

Then we can declare our type:

```
using our_difference_type = Kokkos::detected_or_t<std::ptrdiff_t, diff_t, MyType>;
```

