..
  Use the following convention for headings:

    # with overline, for parts (collections of chapters)

    * with overline, for chapters

    = for sections

    - for subsections

    ^ for subsubsections

    " for paragraphs

..
  Class / method / container name)
  for free functions that are callable, preserve the naming convention, `view_alloc()`

``CoolerView``
==============

.. role:: cpp(code)
    :language: cpp

..
  The (public header) file the user will include in their code

Header File: ``Kokkos_Core.hpp``

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

.. cpp:class:: template <class DataType, class... Traits> CoolerView

  ..
    Template parameters (if applicable)
    Omit template parameters that are just used for specialization/are deduced/ and/or should not be exposed to the user.

  :tparam Foo: Description of the Foo template parameter

  ..
    Parameters (if applicable)

  :param bar: Description of the bar parameter

  .. rubric:: Public Types:

  .. cpp:type:: data_type

    Some interesting description of the type and how to use it.

  .. rubric:: Static Public Member Variables:

  .. cpp:member:: int some_var = 5;

    Description of some_var

    ..
      If you have related info

    .. seealso::

      ..
        We can cross-reference entities

      The :cpp:func:`frobrnicator` free function.

  .. rubric:: Constructors:

  .. cpp:function:: CoolerView(CoolerView&& rhs)

    Whether it's a move/copy/default constructor. Describe what it does.

  ..
    Only include the destructor if it does something interesting as part of the API, such as RAII classes that release a resource on their destructor. Classes that merely
    clean up or destroy their members don't need this member documented.

  .. rubric:: Destructor:

  .. cpp:function:: ~CoolerView()

    Performs some special operation when destroyed.

  .. rubric:: Public Member Functions:

  .. cpp:function:: template<class U> foo(U x)

    Brief description of the function.

    :tparam U: Description of U

    :param: description of x

    ..
      Describe any API changes between versions.

    .. versionchanged:: 3.7.1

      What changed between versions: e.g. Only takes one parameter for foo-style operations instead of two.

  ..
    Use the C++ syntax for deprecation (don't use the Kokkos deprecated macro) as Sphinx will recognize it. We may in the future
    add extra parsing after the html is generated to render this more nicely.

  .. cpp:type:: [[deprecated("in version 4.0.1")]] foobar

    Represents the foobar capability.

    .. deprecated:: 4.0.1

      Use :cpp:type:`foobat` instead.

  .. cpp:type:: foobat

    A better version of foobar.

    .. versionadded:: 4.0.1


Non-Member Functions
--------------------

..
  These should only be listed here if they are closely related. E.g. friend operators. However,
  something like view_alloc shouldn't be here for view

.. cpp:function:: template<class ViewSrc> bool operator==(CoolerView, ViewSrc);

  :tparam ViewDst: the other

  :return: true if :cpp:type:`View::value_type`, :cpp:type:`View::layout_type`, :cpp:type:`View::memory_space`, :cpp:func:`View::rank`, :cpp:func:`View::data()` and :cpp:expr:`View::extent(r)`, for :cpp:expr:`0<=r<rank`, match.

.. cpp:function:: void frobrnicator(CoolerView &v) noexcept

  :param: v the :cpp:class:`CoolerView` to frobnicate

  Frobnicates a CoolerView.

Examples
--------

..
  It may be useful to also have examples for individual functions above.

  Prefer working and compilable examples to prose descriptions (such as "Usage").

.. code-block:: cpp

  #include <Kokkos_Core.hpp>
  #include <cstdio>

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
