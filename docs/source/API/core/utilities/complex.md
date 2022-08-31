# `Kokkos::complex`

Header File: `Kokkos_Core.hpp`

Usage: 
```c++
Kokkos::complex<double> a,b;
a.imag() = 5.0; a.real() = 1.0
b = a;
a += b;
```

## Synopsis 
```c++
template<class Scalar>
class complex {
   
 public:
   typedef Scalar value_type;

 private: 
   value_type re,im;      

 public:

   KOKKOS_INLINE_FUNCTION complex();
   KOKKOS_INLINE_FUNCTION complex(const complex& src);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex(const T& re);
   
   template <class T1, class T2>
   KOKKOS_INLINE_FUNCTION complex(const T1& re, const T2& im)
   
   template<class T>
   complex(const std::complex<T>& src);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const complex<T>& src);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const T& re);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const std::complex<T>& src);
   template<class T>

   operator std::complex<Scalar>() const;

   KOKKOS_INLINE_FUNCTION RealType& imag();
   KOKKOS_INLINE_FUNCTION RealType& real();
   KOKKOS_INLINE_FUNCTION const RealType imag() const;
   KOKKOS_INLINE_FUNCTION const RealType real() const;
   KOKKOS_INLINE_FUNCTION void imag(RealType v);
   KOKKOS_INLINE_FUNCTION void real(RealType v);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator += (const complex<T>& src);
   template<class T>
   complex& operator += (const std::complex<T>& src);
   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator += (const T& real);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator -= (const complex<T>& src);
   template<class T>
   complex& operator -= (const std::complex<T>& src);
   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator -= (const T& real);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator *= (const complex<T>& src);
   template<class T>
   complex& operator *= (const std::complex<T>& src);
   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator *= (const T& real);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator /= (const complex<T>& src);
   template<class T>
   complex& operator /= (const std::complex<T>& src);
   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator /= (const T& real);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator == (const complex<T>& src);
   template<class T>
   complex& operator == (const std::complex<T>& src);
   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator == (const T& real);

   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator != (const complex<T>& src);
   template<class T>
   complex& operator != (const std::complex<T>& src);
   template<class T>
   KOKKOS_INLINE_FUNCTION complex& operator != (const T& real);
};
```

## Public Class Members

### Typedefs
   
 * `value_type`: The scalar type of the real and the imaginary component.

### Constructors
 
 * ```c++
      KOKKOS_INLINE_FUNCTION complex();
   ```
   Default constructor. Initializes the `re` and `im` with `value_type()`.

 * ```c++
      KOKKOS_INLINE_FUNCTION complex(const complex& src);
   ```
   Copy constructor. Sets `re = src.real()` and `im = src.imag()`.
   

 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex(const T& real);
   ```
   Constructor from a real number. Sets `re = real` and `im = value_type()`.
      
 * ```c++
      template <class T1, class T2>
      KOKKOS_INLINE_FUNCTION complex(const T1& real, const T2& imag)
   ```
   Constructor from real numbers. Sets `re = real` and `im = imag`.
      
 * ```c++
      template<class T>
      complex(const std::complex<T>& src);
   ```
   Copy constructor. Sets `re = src.real()` and `im = src.imag()`.

### Assignment and conversion

 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const complex<T>& src);
   ```
   Sets `re = src.real()` and `im = src.imag()`.

 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const T& re);
   ```
   Sets `re = src.real()` and `im = value_type()`.

 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const std::complex<T>& src);
   ```
   Sets `re = src.real()` and `im = src.imag()`.

 * ```c++
      operator std::complex<value_type>() const;
   ```
   Returns `std::complex<value_type>(re,im)`.

### Functions

 * ```c++
      KOKKOS_INLINE_FUNCTION RealType& imag();
   ```
   Return `im`.

 * ```c++
      KOKKOS_INLINE_FUNCTION RealType& real();
   ```
   Return `re`.
 * ```c++
      KOKKOS_INLINE_FUNCTION const RealType imag() const;
   ```
   Return `im`.
 * ```c++
      KOKKOS_INLINE_FUNCTION const RealType real() const;
   ```
   Return `re`.
 * ```c++
      KOKKOS_INLINE_FUNCTION void imag(RealType v);
   ```
   Sets `im = v`.
 * ```c++
      KOKKOS_INLINE_FUNCTION void real(RealType v);
   ```
   Sets `re = v`.

 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator += (const complex<T>& src);
   ```
   Executes `re += src.real(); im += src.imag(); return *this;`

 * ```c++
      template<class T>
      complex& operator += (const std::complex<T>& src);
   ```
   Executes `re += src.real(); im += src.imag(); return *this;`
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator += (const T& real);
   ```
   Executes `re += real; return *this;`
 
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator -= (const complex<T>& src);
   ```
   Executes `re -= src.real(); im -= src.imag(); return *this;`
 * ```c++
      template<class T>
      complex& operator -= (const std::complex<T>& src);
   ```
   Executes `re -= src.real(); im -= src.imag(); return *this;`
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator -= (const T& real);
   ```
   Executes `re -= real; return *this;`

 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator *= (const complex<T>& src);
   ```
   Multiplies the current complex number with the complex number src.
 * ```c++
      template<class T>
      complex& operator *= (const std::complex<T>& src);
   ```
   Multiplies the current complex number with the complex number `src`.
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator *= (const T& real);
   ```
   Executes `re *= real; im *= real; return *this;`
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator /= (const complex<T>& src);
   ```
   Divides the current complex number with the complex number `src`.
 * ```c++
      template<class T>
      complex& operator /= (const std::complex<T>& src);
   ```
   Divides the current complex number with the complex number `src`.
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator /= (const T& real);
   ```
   Executes `re /= real; im /= real; return *this;`
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator == (const complex<T>& src);
   ```
   Returns `re == src.real() && im == src.imag()`.
 * ```c++
      template<class T>
      complex& operator == (const std::complex<T>& src);
   ```
   Returns `re == src.real() && im == src.imag()`.
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator == (const T& real);
   ```
   Returns `re == src.real() && im == value_type()`.
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator != (const complex<T>& src);
   ```
   Returns `re != src.real() || im != src.imag()`.
 * ```c++
      template<class T>
      complex& operator != (const std::complex<T>& src);
   ```
   Returns `re != src.real() || im != src.imag()`.
 * ```c++
      template<class T>
      KOKKOS_INLINE_FUNCTION complex& operator != (const T& real);
   ```
   Returns `re != src.real() || im != value_type()`.
