``ScopeGuard``
==============

.. role::cpp(code)
    :language: cpp

.. role:: cppkokkos(code)
   :language: cppkokkos

Defined in header ``<Kokkos_Core.hpp>``

Usage:

.. code-block:: cpp

    Kokkos::ScopeGuard guard(argc, argv);
    Kokkos::ScopeGuard guard(Kokkos::InitializationSettings()  // (since 3.7)
                                .set_map_device_id_by("random")
                                .set_num_threads(1));


``ScopeGuard`` is a class which ensure thats `Kokkos::initialize <initialize.html#kokkosinitialize>`_ and
`Kokkos::finalize <finalize.html#kokkosfinalize>`_ are called correctly even in the presence of unhandled
exceptions.
It calls `Kokkos::initialize <initialize.html#kokkosinitialize>`_ with the provided arguments in the
constructor and `Kokkos::finalize <finalize.html#kokkosfinalize>`_ in the destructor.


**WARNING: change of behavior in version 3.7**  (see :ref:`note <notes>` below)

Interface
---------

.. cpp:class:: ScopeGuard

    ScopeGuard is a class which ensure thats Kokkos::initialize and Kokkos::finalize are called correctly even in the presence of unhandled exceptions. 


    .. cpp:function:: ScopeGuard(ScopeGuard const&) = delete;

    .. cpp:function:: ScopeGuard(ScopeGuard&&) = delete;

    .. cpp:function:: ScopeGuard& operator=(ScopeGuard const&) = delete;

    .. cpp:function:: ScopeGuard& operator=(ScopeGuard&&) = delete;

    .. cpp:function:: ScopeGuard(int& argc, char* argv[]);

        **until 3.7**

    .. cpp:function:: ScopeGuard(InitArguments const& arguments = InitArguments());
        
        **until 3.7**

    .. cpp:function:: template <class... Args> ScopeGuard(Args&&... args)
        
        **since 3.7**

    .. code-block:: cpp

        template <class... Args>
        ScopeGuard(Args&&... args) {
            // possible implementation
            initialize(std::forward<Args>(args)...);
        }

    .. code-block:: cpp

        ~ScopeGuard() {
            // possible implementation
            finalize();
        }

Parameters
~~~~~~~~~~

* ``argc``: number of command line arguments
* ``argv``: array of character pointers to null-terminated strings storing the command line arguments
* ``arguments``: ``struct`` object with valid initialization arguments
* ``args``: arguments to pass to `Kokkos::initialize <initialize.html#kokkosinitialize>`_

Note that all of the parameters above are passed to the ``Kokkos::initialize`` called internally.  See `Kokkos::initialize <initialize.html#kokkosinitialize>`_ for more details.


.. _notes:

Notes
~~~~~
Since Kokkos version 3.7, ``ScopeGuard`` unconditionally forwards the provided
arguments to `Kokkos::initialize <initialize.html#kokkosinitialize>`_, which means they have the same
preconditions.  Until version 3.7, ``ScopeGuard`` was calling
``Kokkos::initialize`` in its constructor only if ``Kokkos::is_initialized()`` was
``false``, and it was calling ``Kokkos::finalize`` in its destructor only if it
called ``Kokkos::initialize`` in its constructor.

We dropped support for the old behavior.  If you think you really need it, you
may do:

.. code-block:: cpp

    auto guard = std::unique_ptr<Kokkos::ScopeGuard>(
        Kokkos::is_initialized() ? new Kokkos::ScopeGuard() : nullptr);

or

.. code-block:: cpp

    auto guard = Kokkos::is_initialized() ? std::make_optional<Kokkos::ScopeGuard>()
                                        : std::nullopt;

with C++17.  This will work regardless of the Kokkos version.

Example
~~~~~~~

.. code-block:: cpp

    int main(int argc, char* argv[]) {
        Kokkos::ScopeGuard guard(argc, argv);
        Kokkos::View<double*> my_view("my_view", 10);
        // my_view destructor called before Kokkos::finalize
        // ScopeGuard destructor called, calls Kokkos::finalize
    }


See also
~~~~~~~~
* `Kokkos::initialize <initialize.html#kokkosinitialize>`_
* `Kokkos::finalize <finalize.html#kokkosfinalize>`_
