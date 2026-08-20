Random-Number
=============

.. role:: cpp(code)
    :language: cpp

Rand
----

Header Files: ``<Kokkos_Core.hpp>``, ``<Kokkos_Random.hpp>``

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

Header Files: ``<Kokkos_Core.hpp>`` ``<Kokkos_Random.hpp>``

Synopsis
--------

Kokkos_Random provides the structure necessary for pseudorandom number generators.
Kokkos currently ships two families of generators:

* the XorShift generators (``Random_XorShift64_Pool``, ``Random_XorShift1024_Pool``),
  based on Vigna, Sebastiano (2014). [*"An experimental exploration of
  Marsaglia's xorshift generators, scrambled." See:
  http://arxiv.org/abs/1402.6246*];
* ``Random_SFC64_Pool``, implementing Chris Doty-Humphrey's Small Fast
  Counting (SFC64) generator (see the :ref:`dedicated section
  <random_sfc64_pool>` below for its distinguishing properties).

All of these share the same ``Pool``/``Generator`` interface described in
this page, and can be used interchangeably.

The Random number generators themselves have two components:
a state-pool and the actual generator. A state-pool manages
a number of generators so that each active thread is able
to grab its own. This allows the generation of random numbers
which are independent between threads. Note that in contrast
to **CuRAND**, none of the functions of the pool (or the generator)
are collectives, i.e. all functions can be called inside conditionals.

.. code-block:: cpp

    template<class DeviceType>
    class Pool {
      public:

      using device_type = DeviceType;
      using generator_type = Generator<DeviceType>;

      Pool();
      Pool(uint64_t seed);
      Pool(uint64_t seed, uint64_t num_states);
      Pool(const typename DeviceType::execution_space& exec, uint64_t seed);
      Pool(const typename DeviceType::execution_space& exec, uint64_t seed, uint64_t num_states);

      void init(uint64_t seed, uint64_t num_states);  // deprecated since Kokkos 5.0
      generator_type get_state();
      void free_state(generator_type Gen);
    }

Construction and Initialization
--------------------------------

A Pool of Generators is initialized using a starting seed and establishing
a pool_size of num_states. This initialization process is always platform
independent and deterministic, but its execution differs depending on the
underlying generator family:

* For the XorShift generators, a single ``Random_XorShift64`` generator is
  run **serially** (on the host) to seed every state in the pool one after
  the other, one state's seed being derived from the previous one's.
* For ``Random_SFC64_Pool``, each stream's initial state depends only on
  the pair ``(seed, stream index)``, so the pool's states can instead be
  initialized **in parallel** by the target backend when one is available
  — see :ref:`Parallel initialization <random_sfc64_pool>` below. This can
  be noticeably faster for large pools.

In both cases, requesting a generator locks its state, guaranteeing that
each thread has a private (independent) generator. (Note, getting a state
on a Cuda device involves atomics, making it non-deterministic!) Upon
completion, a generator is returned to the state pool, unlocking it, and
upon updating of its status, once again becomes available within the pool.

Pool constructors that do not take an execution space instance are synchronous, and use the default execution space instance of the provided `DeviceType`.
Pool constructors that take an execution space instance are asynchronous.

Use
---

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

.. _random_sfc64_pool:

Random_SFC64_Pool
------------------

Header Files: ``<Kokkos_Core.hpp>``, ``<Kokkos_Random.hpp>``

``Random_SFC64_Pool`` is an alternative pool/generator pair implementing
Chris Doty-Humphrey's Small Fast Counting (SFC64) pseudorandom number
generator, released into the public domain. (Note: this generator is
sometimes referred to as "Small Fast Chaotic", e.g. in NumPy's
documentation)
It follows the same ``Pool``/``Generator`` interface described above
(``get_state()``, ``free_state()``, etc.), and can be used as a drop-in
replacement for ``Random_XorShift64_Pool``.

Unlike the XorShift generators, SFC64 embeds a 64-bit counter in its
internal state. From a single 64-bit seed, this gives access to
2\ :sup:`64`\  independent streams, each with a period of at least
2\ :sup:`64`\  (with an expected period on the order of 2\ :sup:`255`\ ).

.. code-block:: cpp

  template<class DeviceType>
  class Random_SFC64_Pool {
    public:

    using device_type = DeviceType;
    using generator_type = Random_SFC64<DeviceType>;

    Random_SFC64_Pool();
    Random_SFC64_Pool(uint64_t seed);
    Random_SFC64_Pool(uint64_t seed, uint64_t num_states);
    // Useful in distributed settings to be reproducible
    Random_SFC64_Pool(uint64_t seed, uint64_t seed_offset, uint64_t num_states);

    // Asynchronous constructors :
    Random_SFC64_Pool(const execution_space& exec, uint64_t seed);
    Random_SFC64_Pool(const execution_space& exec, uint64_t seed, uint64_t num_states);
    Random_SFC64_Pool(const execution_space& exec, uint64_t seed,
                       uint64_t seed_offset, uint64_t num_states);

    generator_type get_state();
    void free_state(generator_type gen);
  }

Parallel Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

Because each SFC64 stream can be seeded independently from its index
(seed, counter offset), the pool's states can be initialized concurrently
by the target backend. This is in contrast to ``Random_XorShift64_Pool``
and ``Random_XorShift1024_Pool``, whose states are chained from one
another and must therefore be initialized serially. When a parallel
backend is enabled, constructing a ``Random_SFC64_Pool`` will initialize
its states in parallel, which can be noticeably faster for large pools.

Reproducible Stream Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because each stream is uniquely determined by ``(seed, stream_index)``,
splitting a computation across multiple pools (for instance one per MPI
rank) while preserving bit-for-bit reproducibility only requires giving
every pool the same seed and a ``seed_offset`` equal to the number of
streams already handed out to previous pools. For example, requesting
1000 states from a single pool is equivalent to requesting 500 states
from one pool with ``seed_offset = 0`` and 500 states from a second pool
with the same seed and ``seed_offset = 500``: both setups produce the
exact same 1000 streams.

.. code-block:: cpp

  #include <Kokkos_Core.hpp>
  #include <Kokkos_Random.hpp>

  int main(int argc, char *argv[]) {
      Kokkos::ScopeGuard guard(argc, argv);

      uint64_t seed = 12345;

      // Pool 0 handles streams [0, 500), pool 1 handles streams [500, 1000)
      // — identical results to a single pool of 1000 states with seed_offset 0.
      Kokkos::Random_SFC64_Pool<> pool0(seed, /*seed_offset=*/0, /*num_states=*/500);
      Kokkos::Random_SFC64_Pool<> pool1(seed, /*seed_offset=*/500, /*num_states=*/500);

      // ... use pool0 / pool1 exactly like Random_XorShift64_Pool
  }

Use
~~~

As with the other pools, it is highly recommended to use the generic
``Kokkos::rand`` struct to generate numbers of specific types or within
specific distributions, rather than relying on legacy methods attached
to the generator object itself.

.. code-block:: cpp

  #include <Kokkos_Core.hpp>
  #include <Kokkos_Random.hpp>

  int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
          // 1. Initialize the SFC64 generator pool with a seed
          uint64_t seed = 123456789;
          Kokkos::Random_SFC64_Pool<Kokkos::DefaultExecutionSpace> rand_pool(seed);

          int N = 1000;
          Kokkos::View<double*> random_numbers("random_numbers", N);

          // 2. Use the pool in a parallel kernel
          Kokkos::parallel_for("GenerateRandomNumbers", N, KOKKOS_LAMBDA(const int i) {
              // Get a state/generator from the pool
              auto generator = rand_pool.get_state();

              // Generate a random double (e.g., between 0.0 and 1.0)
              // Recommended API using the Kokkos::rand struct:
              random_numbers(i) = Kokkos::rand<decltype(generator), double>::draw(generator);

              // Alternatively, generate a number in a specific range [min, max)
              double val_range = Kokkos::rand<decltype(generator), double>::draw(generator, 10.0, 20.0);

              // Return the state back to the pool to avoid deadlocks
              rand_pool.free_state(generator);
          });
      }
      Kokkos::finalize();
      return 0;
  }
