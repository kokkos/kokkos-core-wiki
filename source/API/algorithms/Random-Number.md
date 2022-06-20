# Random-Number

#### headers:  Kokkos_Core.hpp, Kokkos_Complex.hpp

`template<class Generator>`
`struct rand<Generator, gen_data_type>`

 * `KOKKOS_INLINE_FUNCTION
    static gen_func_type max(){return type_value}`

 * `KOKKOS_INLINE_FUNCTION
    static gen_func_type draw(Generator& gen)  {return gen_data_type((gen.rand()&gen_return_value)}`

 * `KOKKOS_INLINE_FUNCTION
    static gen_func_type draw(Generator& gen, const gen_data_type& range)  {return gen_data_type((gen.rand(range));}`

 * `KOKKOS_INLINE_FUNCTION
    static gen_func_type draw(Generator& gen, const gen_data_type& start, const gen_data_type& end)
                     {return gen_data_type(gen.rand(start,end));}`


Function specializations for _gen_data_type_, _gen_func_type_ and _type_value_

All functions and classes listed here are part of the `Kokkos::` namespace. 

|gen_data_type |gen_func_type | type_value | gen_return_value            |
|:-------------|:-------------|:-----------|:----------------------------|
| char | short | 127 | (&0xff+256)%256 |
| short | short | 32767 | (&0xffff+65536)%32768  |
| int | int  | MAX_RAND |  ? |
| uint | uint | MAX_URAND |  ? |
| long | long | MAX_RAND or MAX_RAND64 |  ? |
| ulong | ulong  | MAX_RAND or MAX_RAND64 |  ? |
| long long | long long  | MAX_RAND64 |  ? |
| ulong long | ulong long  | MAX_URAND64 |  ? |
| float | float  | 1.0f |  ? |
| double | double  | 1.0 |  ? |
| complex<float> | complex<float>  | 1.0,1.0 |  ? |
| complex<double> | complex<double>  | 1.0,1.0 |  ? |
|  |  |  |  |

where the maximum values of the XorShift function values are given by the following enums.
*   enum {MAX_URAND = 0xffffffffU};
*   enum {MAX_URAND64 = 0xffffffffffffffffULL-1};
*   enum {MAX_RAND = static_cast<int>(0xffffffffU/2)};
*   enum {MAX_RAND64 = static_cast<int64_t>(0xffffffffffffffffULL/2-1)};

# Kokkos::Generator


Header Files:  `Kokkos_core.hpp`
               `Kokkos_Complex.hpp`

## Synopsis
Kokkos_Random provides the structure necessary for 
pseudorandom number generators. These generators are
based on Vigna, Sebastiano (2014). ["_An_
_experimental exploration of Marsaglia's xorshift generators,_
_scrambled."  See: http://arxiv.org/abs/1402.6246_] .

**The text that had been posted here is incomplete and has been removed until it is completed.**
