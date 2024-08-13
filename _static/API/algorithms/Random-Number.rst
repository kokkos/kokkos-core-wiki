Random-Number
=============

.. role:: cppkokkos(code)
    :language: cppkokkos

Rand
----

Header Files: ``<Kokkos_core.hpp>``, ``<Kokkos_Complex.hpp>``

.. code-block:: cpp

   template<class Generator>
   struct rand<Generator, gen_data_type>
   {
     KOKKOS_INLINE_FUNCTION
     static gen_func_type max(){
       return type_value;
     }

     KOKKOS_INLINE_FUNCTION
     static gen_func_type draw(Generator& gen){
       return gen_data_type((gen.rand()&gen_return_value)
     }

     KOKKOS_INLINE_FUNCTION
     static gen_func_type draw(Generator& gen,
                               const gen_data_type& range){
       return gen_data_type((gen.rand(range));
     }

     KOKKOS_INLINE_FUNCTION
     static gen_func_type draw(Generator& gen,
                               const gen_data_type& start,
			       const gen_data_type& end){
       return gen_data_type(gen.rand(start,end));
     }

Function specializations for ``gen_data_type``, ``gen_func_type`` and ``type_value``.
All functions and classes listed here are part of the ``Kokkos::`` namespace.

+-------------------+-------------------+---------------------------+-----------------------+
| gen_data_type     | gen_func_type     | type_value                | gen_return_value      |
+===================+===================+===========================+=======================+
| char              | short             | 127                       | (&0xff+256)%256       |
+-------------------+-------------------+---------------------------+-----------------------+
| short             | short             | 32767                     | (&0xffff+65536)%32768 |
+-------------------+-------------------+---------------------------+-----------------------+
| int               | int               | MAX_RAND                  |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| uint              | uint              | MAX_URAND                 |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| long              | long              | MAX_RAND or MAX_RAND64    |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| ulong             | ulong             | MAX_RAND or MAX_RAND64    |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| long long         | long long         | MAX_RAND64                |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| ulong long        | ulong long        | MAX_URAND64               |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| float             | float             | 1.0f                      |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| double            | double            | 1.0                       |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| complex<float>    | complex<float>    | 1.0,1.0                   |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+
| complex<double>   | complex<double>   | 1.0,1.0                   |  ?                    |
+-------------------+-------------------+---------------------------+-----------------------+

where the maximum values of the XorShift function values are given by the following enums.

* enum {MAX_URAND = 0xffffffffU};
* enum {MAX_URAND64 = 0xffffffffffffffffULL-1};
* enum {MAX_RAND = static_cast<int>(0xffffffffU/2)};
* enum {MAX_RAND64 = static_cast<int64_t>(0xffffffffffffffffULL/2-1)};

Generator
=========

Header Files: ``<Kokkos_core.hpp>`` ``<Kokkos_Complex.hpp>``

Synopsis
--------

Kokkos_Random provides the structure necessary for pseudorandom number generators. These generators are based on Vigna, Sebastiano (2014). [*"An experimental exploration of Marsaglia's xorshift generators, scrambled." See: http://arxiv.org/abs/1402.6246*].

The Random number generators themselves have two components:
a state-pool and the actual generator. A state-pool manages
a number of generators so that each active thread is able
to grab its own. This allows the generation of random numbers
which are independent between threads. Note that in contrast
to **CuRAND**, none of the functions of the pool (or the generator)
are collectives, i.e. all functions can be called inside conditionals.

.. code-block:: cpp

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

A Pool of Generators are initialized using a starting seed and establishing
a pool_size of num_states. The Random_XorShift64 generator is used in serial
to initialize all states making the initialization process platform independent
and deterministic. Requesting a generator locks it's state guaranteeing that
each thread has a private (independent) generator. (Note, getting a state on a Cuda
device involves atomics, making it non-deterministic!)
Upon completion, a generator is returned to the state pool, unlocking
it, and upon updating of it's status, once again becomes available
within the pool.

Given a pool and selection of a generator from within that pool,
the next step is development of a functor that will draw random
numbers, of the desired type, using the generator.

.. code-block:: cpp

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

      //Draw a equidistributed uint32_t in the range [0,MAX_URAND)
      KOKKOS_INLINE_FUNCTION
      uint32_t urand();

      //Draw a equidistributed uint32_t in the range [0,range)
      KOKKOS_INLINE_FUNCTION
      uint32_t urand(const uint32_t& range);

      //Draw a equidistributed uint32_t in the range [start,end)
      KOKKOS_INLINE_FUNCTION
      uint32_t urand(const uint32_t& start, const uint32_t& end );
    }

For the selected 32-bit unsigned integer type, three range options are shown: [0,MAX_URAND), [0,range) and [start,end).
The first, and default, option selects unsigned integers over max possible range for that data type. The defined value of MAX_URAND is shown above as an enum. (And also shown is maX_URAND for a 64-bit unsigned integer.) The latter two options cover a user-defined range of integers.

More for other data types: Scalar, uint64_t, int, int32_t, int64_t, float, double; also normal distribution and a View-fill option for the [0, range) and [start, end) options.

Example
-------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <Kokkos_Random.hpp>

    int main(int argc, char *argv[]) {
        Kokkos::ScopeGuard guard(argc, argv);

        Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

        int total = 1000000;
        int count;
        Kokkos::parallel_reduce(
            "approximate_pi", total,
            KOKKOS_LAMBDA(int, int& local_count) {
                // acquire the state of the random number generator engine
                auto generator = random_pool.get_state();

                double x = generator.drand(0., 1.);
                double y = generator.drand(0., 1.);

                // do not forget to release the state of the engine
                random_pool.free_state(generator);

                if (x * x + y * y <= 1.) {
                    ++local_count;
                }
            },
            count);

        printf("pi = %f\n", 4. * count / total);
    }
