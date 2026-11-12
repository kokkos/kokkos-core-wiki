``BadAlloc``
============

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. warning::

   ``BadAlloc`` is in the ``Kokkos::Experimental`` namespace and may therefore
   change in future releases without prior notice.

Usage
-----

.. code-block:: cpp

    try {
      void* ptr = Kokkos::kokkos_malloc(huge_size);
    } catch (Kokkos::Experimental::BadAlloc const& e) {
      std::cerr << "Allocation failed: " << e.what() << '\n';
    }

Exception type thrown when Kokkos fails to allocate memory, for example from
:cpp:func:`kokkos_malloc`, :cpp:func:`kokkos_realloc` or when allocating a
:cpp:class:`View`.

Interface
---------

.. cpp:class:: Experimental::BadAlloc : public std::runtime_error

   .. versionadded:: 5.2

Notes
-----

Version History
^^^^^^^^^^^^^^^
* Before Kokkos 4.4, allocation failures threw backend-specific types such as
  ``Experimental::RawMemoryAllocationFailure``. These types no longer exist.
* From Kokkos 4.4 up to 5.1, allocation failures throw a ``std::runtime_error``.
* Starting with Kokkos 5.2, allocation failures throw ``Experimental::BadAlloc``.
  Since it derives from ``std::runtime_error``, code catching
  ``std::runtime_error`` keeps working.
