# Random Number Generators

This section documents the basis for the Kokkos Random Number Generators and outlines their usage.

## Description of Generators

The implementation of random number generators in Kokkos is based on two primary references (Marsaglia<sup>1</sup>, 2003; Vigna<sup>2</sup>, 2014). Marsaglia's paper introduces **Xorshift** (bit-shifts with exclusive ors) pseudorandom number generators and tests some non-linear transformations to improve their performance on standard test suites. In spite of less than desirable performance numbers statistically, he showed that these generators are **very fast** and can generate hundreds of millions of random numbers in very few clock cycles. Using the same 64-bit state as Marsaglia, Vigna made improved parameter selections than Marsaglia but still achieve "poor" results. Applying an invertible multiplication step (hinted at by Marsaglia and called a "scramble" by Vigna) and correcting/adding to Marsaglia's xorshift64 algorithms, much improved behavior on test suites and higher statistical quality was achieved.   Vigna describes tests on 1024 and 4096-bit versions of these generators later in his experimental process.

Kokkos implements a 64-bit and 1024-bit (high-dimensional) version of the scrambled xorshift generators. Please      refer to the above references for details of the algorithms.

***
<sup>1</sup> George Marsaglia. 2003. _XorshiftRNGs._ Journal of Statistical Software 8, 14 (2003), 1-6. 

<sup>2</sup> Vigna, Sebastiano (2014). "_An experimental exploration of Marsaglia's xorshift generators, scrambled_“ ( http://arxiv.org/abs/1402.6246)
***

## Application of Random Number Generators

The Process, the Steps