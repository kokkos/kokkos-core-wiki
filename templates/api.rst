..
  Use the following convention for headings:

    # with overline, for parts

    * with overline, for chapters

    = for sections

    - for subsections

    ^ for subsubsections

    " for paragraphs

..
  Class / method / container name)
  for free functions that are callable, preserve the naming convention, `view_alloc()`

``{{ entity_name }}``
=====================

.. role:: cppkokkos(code)
    :language: cppkokkos

..
  The (pulic header) file the user will include in their code

Header File: ``{{ core_header_file }}``

..
  High-level, human-language summary of what the thing does, and if possible, brief statement about why it exists (2 - 3 sentences, max);

Description
-----------

..
  The API of the entity.
Interface
---------

..
  The declaration or signature of the entity.

.. cpp:class:: template <class DataType, class... Traits> View

  ..
    Template parameters (if applicable)
    Omit template parameters that are just used for specialization/is deduced/ and/or should not be exposed to the user.

  .. rubric:: Template Parameters

  :tparam Foo: Description of the Foo template parameter

  ..
    Parameters (if applicable)

  .. rubric:: Parameters

  :param bar: Description of the bar parameter

  .. rubric:: Public Types

  .. cppkokkos:type:: data_type

    Some interesting description of the type and how to use it.

  .. rubric:: Static Public Member Variables

  .. cppkokkos:member:: some_var

    Description of some_var

    ..
      If you have related info

    .. seealso::

      The :func:`frobrnicator` free function.

  .. rubric:: Constructor

  .. cppkokkos:func:: View(View&& rhs)

    Whether it's a move/copy/default constructor. Describe what it does.

  ..
    Only include the destructor if it does something interesting as part of the API. This usually not the case.

  .. rubric:: Destructor

  .. cppkokkos:func:: ~View()

    Document what special effect the destructor has.

  .. rubric:: Public Member Functions

  .. cppkokkos:func:: template<class U> foo(U x)

    Brief description of the function.

    :tparam U: Description of U

    :param: description of x

Non-Member Functions
----------------------

..
  These should only be listed here if they are closely related. E.g. friend operators. However,
  something like view_alloc shouldn't be here for view

.. cppkokkos:function:: template<class ViewDst, class ViewSrc> bool operator==(ViewDst, ViewSrc);

    :tparam ViewDst: the first view type
    :tparam ViewSrc: the second view type

    :return: true if :cppkokkos:type:`~View::value_type`, :cppkokkos:type:`~View::array_layout`, :cppkokkos:any:`~View::memory_space`, :cppkokkos:any:`~View::rank`, :cppkokkos:any:`~View::data()` and :cppkokkos:any:`~View::extent` (r), for :code:`0<=r<rank`, match.


Examples
--------

..
  It may be useful to also have examples for individual functions above.

  Prefer examples to usage.

.. code-block:: cpp

  #include<Kokkos_Core.hpp>
  #include<cstdio>

  int main(int argc, char* argv[]) {
     Kokkos::initialize(argc,argv);

     int N0 = atoi(argv[1]);
     int N1 = atoi(argv[2]);

     Kokkos::View<double*> a("A",N0);
     Kokkos::View<double*> b("B",N1);

     Kokkos::parallel_for("InitA", N0, KOKKOS_LAMBDA (const int& i) {
       a(i) = i;
     });

     Kokkos::parallel_for("InitB", N1, KOKKOS_LAMBDA (const int& i) {
       b(i) = i;
     });

     Kokkos::View<double**,Kokkos::LayoutLeft> c("C",N0,N1);
     {
       Kokkos::View<const double*> const_a(a);
       Kokkos::View<const double*> const_b(b);
       Kokkos::parallel_for("SetC", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{N0,N1}),
         KOKKOS_LAMBDA (const int& i0, const int& i1) {
         c(i0,i1) = a(i0) * b(i1);
       });
     }

     Kokkos::finalize();
  }
