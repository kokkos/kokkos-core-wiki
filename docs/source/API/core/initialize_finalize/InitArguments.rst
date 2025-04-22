InitArguments
=============

.. role:: cpp(code)
   :language: cpp

.. _KokkosInitialize: initialize.html
.. |KokkosInitialize| replace:: ``Kokkos::initialize``

.. _KokkosInitializationSetting: InitializationSettings.html
.. |KokkosInitializationSetting| replace:: ``Kokkos::InitializationSettings``

Defined in ``<Kokkos_Core.hpp>`` header.

.. warning:: Deprecated since 3.7, removed in 4.3, **use** ``Kokkos::InitializationSettings`` **instead**

Interface
---------

.. cpp:struct:: InitArguments

   .. cpp:member:: int num_threads

   .. cpp:member:: int num_numa

   .. cpp:member:: int device_id

   .. cpp:member:: int ndevices

   .. cpp:member:: int skip_device

   .. cpp:member:: bool disable_warnings

   .. cpp:function:: InitArguments()

``InitArguments`` is a struct that can be used to programmatically define the arguments passed to |KokkosInitialize|_. It was deprecated in version 3.7 in favor of |KokkosInitializationSetting|_.

One of the main reasons for replacing it was that user-specified data members cannot be distinguished from defaulted ones.

Example
~~~~~~~

.. code-block:: cpp

   #include <Kokkos_Core.hpp>

   int main() {
     Kokkos::InitArguments arguments;
     arguments.num_threads = 2;
     arguments.device_id = 1;
     arguments.disable_warnings = true;
     Kokkos::initialize(arguments);
     // ...
     Kokkos::finalize();
   }


See also
~~~~~~~~

* |KokkosInitializationSetting|_
* |KokkosInitialize|_
