# Kokkos: The C++ Performance Portability Programming Model

Kokkos implements a programming model in C++ for writing performance portable
applications targeting all major HPC platforms. For that purpose it provides
abstractions for both parallel execution of code and data management.
Kokkos is designed to target complex node architectures with N-level memory
hierarchies and multiple types of execution resources. It currently can use
CUDA, HIP, SYCL, HPX, OpenMP, and C++ threads as backend programming models,
with several other backends in development.

This Wiki provides a number of resources for both new and old Kokkos developers.
```
Note: The Wiki is a work in progress. 
      Some of the content is not yet available (red links).
```
## **[[Deprecated Functionality|DeprecationPage]]**
List of deprecated features, their replacements and reason for deprecation.

## **[[Upcoming and Past Events|EventsPage]]**

## **[[The Programming Guide|The Kokkos Programming Guide]]**

The Programming Guide contains in-depth descriptions of the capabilities provided
by Kokkos. This includes explanation of design decisions as well as best practice
hints. This is a good place to start off your Kokkos journey other than attending 
one of our Tutorial days. 

## **[[The API Reference|API-Reference]]**

This is the place to go for a fast lookup of syntax. Developers who have previously 
worked a lot with other shared memory models such as OpenMP, CUDA, or OpenCL may be
able to simply look up how certain capabilities are used in Kokkos

<!--- Kokkos Documentation "how to" guides will link to README.md files (via
the kokkos, etc. user) from the main project repos, i.e., Kokkos, Kokkos
Kernels, Kokkos Tools.  For development and testing purposes, use the
ajpowelsnl user, but update the link after merging into `main`.
 --->

## [Kokkos Documentation "How To" Guide](https://github.com/kokkos/kokkos/wiki/README)

# [The Kokkos Eco-System](https://github.com/kokkos)

The Kokkos Programming Model is not the only resource available. There are a number 
of projects which can help HPC developers in their work.

## [Kokkos-Tutorials](https://github.com/kokkos/kokkos-tutorials)

This project has extensive Tutorials for Kokkos including hands-on exercises.
New Kokkos developers, even with little to no previous parallel programming experience
will be taken through the basics of using Kokkos to parallelize applications.
There is also a tutorial available for learning the basics of profiling. 

## [Kokkos-Tools](https://github.com/kokkos/kokkos-tools)

Kokkos Tools provide profiling and debugging capabilities which access built-in 
instrumentation of Kokkos. They make it significantly easier to understand what is 
going on in a large Kokkos application and thus help you to find errors and performance
issues. 

## [Kokkos-Kernels](https://github.com/kokkos/kokkos-kernels)

Many if not most high performance computing applications rely on dense and sparse BLAS 
capabilities or graph kernels. Kokkos-Kernels aims at providing optimized kernels and
interfacing to vendor libraries through a Kokkos View based interface. No need to figure
out what the dimensions, strides and memory spaces of your data structures are, 
Kokkos Views know about that.   
 
