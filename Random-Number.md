# Kokkos::Generator


Header Files:  `Kokkos_core.hpp`
               `Kokkos_Complex.hpp`

## Synopsis
Kokkos_Random provides the structure necessary for 
pseudorandom number generators. These generators are
based on Vigna, Sebastiano (2014). ["_An_
_experimental exploration of Marsaglia's xorshift generators,_
_scrambled."  See: http://arxiv.org/abs/1402.6246_] .

The Random number generators themselves have two components: 
a state-pool and the actual generator. A state-pool manages 
a number of generators so that each active thread is able 
to grab its own. This allows the generation of random numbers 
which are independent between threads. Note that in contrast 
to **CuRAND**, none of the functions of the pool (or the generator) 
are collectives, i.e. all functions can be called inside conditionals.

```c++
 template<class Device>
 class Pool {
   public:
   typedef Device Device_Type;
   typedef Generator<Device> Generator_Type;
 
   Pool();
   Pool(RanNum_type seed);

   void init(RanNum_type seed, int num_states);
   Generator_Type get_state();
   void free_state(Generator_Type Gen);
 }
```
A Pool of Generators are intialized using a starting seed and establishing 
a pool_size of num_states. The Random_XorShift64 generator is used in serial 
to initialize all states making the intialization process platform independent 
and deterministic. Requesting a generator locks it's state guaranteeing that
each thread has a private (independent) generator. (Note, getting a state on a Cuda
device involves atomics, making it non-deterministic!)
Upon completion, a generator is returned to the state pool, unlocking
it, and upon updating of it's status, once again becomes available
within the pool.

Given a pool and selection of a generator from within that pool,
the next step is development of a functor that will draw random
numbers, of the desired type, using the generator.

```c++
    template<class Device>
    class Generator {
     public:

    typedef DeviceType device_type;

    //Max return values of respective [X]rand[S]() functions (XorShift).
    enum {MAX_URAND = 0xffffffffU};
    enum {MAX_URAND64 = 0xffffffffffffffffULL-1};
    enum {MAX_RAND = static_cast<int>(0xffffffffU/2)};
    enum {MAX_RAND64 = static_cast<int64_t>(0xffffffffffffffffULL/2-1)};

    //Init with a state and the idx with respect to pool. Note: in serial the
    //Generator can be used by just giving it the necessary state arguments
    KOKKOS_INLINE_FUNCTION
    Generator (STATE_ARGUMENTS, int state_idx = 0);

    //Draw a equidistributed uint32_t in the range (0,MAX_URAND]
    KOKKOS_INLINE_FUNCTION
    uint32_t urand();

    //Draw a equidistributed uint32_t in the range (0,range]
    KOKKOS_INLINE_FUNCTION
    uint32_t urand(const uint32_t& range);

    //Draw a equidistributed uint32_t in the range (start,end]
    KOKKOS_INLINE_FUNCTION
    uint32_t urand(const uint32_t& start, const uint32_t& end );
    }
```

For the selected 32-bit unsigned integer type, three range options are shown: [0,MAX_URAND], [0,range] and [range].
The first, and default, option selects unsigned integers over max possible range for that data type. The defined value of MAX_URAND is shown above as an enum. (And also shown is maX_URAND for a 64-bit unsigned integer.) The latter two options cover a user-defined range of integers.

More for other data types: Scalar, uint64_t, int, int32_t, int64_t, float, double; also normal distribution and a View-fill option for the [0, range] and [start, end] options.
