# Chapter 5

# Initialization

In order to use Kokkos an initialization call is required. That call is responsible for acquiring hardware resources such as threads. Typically, this call should be placed right at the start of a program. If you use both MPI and Kokkos, your program should initialize Kokkos right after calling `MPI_Init`. That way, if MPI sets up process binding masks, Kokkos will get that information and use it for best performance. Your program must also _finalize_ Kokkos when done using it in order to free hardware resources.

## Initialization by command-line arguments

The simplest way to initialize Kokkos is by calling the following function:

    Kokkos::initialize(int& argc, char* argv[]); 

Just like `MPI_Init`, this function interprets command-line arguments to determine the requested settings. Also like `MPI_Init`, it reserves the right to remove command-line arguments from the input list. This is why it takes `argc` by reference, rather than by value; it may change the value on output.

This function will initialize the default execution space

    Kokkos::DefaultExecutionSpace;

and its default host execution space

    Kokkos::DefaultHostExecutionSpace;

if applicable. These defaults depend on the Kokkos configuration. Kokkos chooses the two spaces using the following list, ordered from low to high:

1. `Kokkos::Serial` 
1. `Kokkos::Threads`
1. `Kokkos::OpenMP`
1. `Kokkos::Cuda`

The highest execution space in the list which is actually enabled is Kokkos' default execution space, and the highest enabled host execution space is Kokkos' default host execution space. (Currently, the only non-host execution space is `Cuda`.) For example, if  `Kokkos::Cuda`, `Kokkos::OpenMP`, and `Kokkos::Serial` are enabled, then `Kokkos::Cuda` is the
default execution space and `Kokkos::OpenMP` is the default host execution space.\footnote{This is the preferred set of defaults when CUDA and OpenMP are enabled. If you use a thread-parallel host execution space, we prefer Kokkos' OpenMP back-end, as this ensures compatibility of Kokkos' threads with the application's direct use of OpenMP threads. Kokkos cannot promise that its Threads back-end will not conflict with the application's direct use of operating system threads.}

Command-line arguments come in ``prefixed'' and ``non-prefixed'' versions. Prefixed versions start with the string \verb!--kokkos-!. `Kokkos::initialize` will remove prefixed options from the input list, but will preserve non-prefixed options. Argument options are given with an equals (\verb!=!) sign. If the same argument occurs more than once, the last one counts. Furthermore, prefixed versions of the command line arguments take precedence over the non-prefixed ones. For example, the arguments
\begin{verbatim}
--kokkos-threads=4 --threads=2
\end{verbatim}
set the number of threads to 4, while
\begin{verbatim}
--kokkos-threads=4 --threads=2 --kokkos-threads=3
\end{verbatim}
set the number of threads to 3.