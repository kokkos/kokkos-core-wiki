The concept of an `ExecutionSpace` is the fundamental abstraction to represent the "where" and the "how" that execution takes place in Kokkos.  Most code that uses Kokkos should be written to the *generic concept* of an `ExecutionSpace` rather than any specific instance.  This page talks practically about how to *use* the common features of execution spaces in Kokkos; for a more formal and theoretical treatment, see [this document](Kokkos Concepts).

> *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates in C++ has used concepts whether they knew it or not.  Please do not be confused by the word "concept" itself, which is now more often associated with a shiny new C++20 language feature.  Here, "concept" just means "what you're allowed to do with a type that is a template parameter in certain places".


