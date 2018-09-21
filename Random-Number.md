
#### headers:  Kokkos_Core.hpp, Kokkos_Complex.hpp

`template<class Generator>`
`struct rand<Generator, gen_data_type>`

 * `KOKKOS_INLINE_FUNCTION
    static gen_func_type max(){return type_value}`

 * `KOKKOS_INLINE_FUNCTION
    static gen_func_type draw(Generator& gen)  {return gen_data_type((gen.rand()&gen_return__value)}`

 * `KOKKOS_INLINE_FUNCTION
    static gen_func_type draw(Generator& gen, const gen_data_type& range)  {return gen_data_type((gen.rand(range));}`

 * `KOKKOS_INLINE_FUNCTION
    static gen_func_type draw(Generator& gen, const gen_data_type& start, const gen_data_type& end)
                     {return gen_data_type(gen.rand(start,end));}`


Function specializations for _gen_data_type_, _gen_func_type_ and _type_value_
*gen_data_type:* Scalar, char, short, int, uint, long, ulong, long long, ulong long, float, double, complex<float>, complex<double>
