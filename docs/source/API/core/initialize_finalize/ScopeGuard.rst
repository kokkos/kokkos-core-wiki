``ScopeGuard``
==============

.. role:: cppkokkos(code)
   :language: cppkokkos

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

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


.. warning:: Change of behavior in version 3.7 (see below)

Description
-----------

.. cpp:class:: ScopeGuard

    A class ensuring that ``Kokkos::initialize`` and ``Kokkos::finalize`` are called correctly even in the presence of unhandled exceptions.

    .. rubric:: Constructors

    .. cpp:function:: ScopeGuard(int& argc, char* argv[]);

       :param argc: number of command line arguments
       :param argv: array of character pointers to null-terminated strings storing the command line arguments

       .. warning:: Valid until 3.7

    .. cpp:function:: ScopeGuard(InitArguments const& arguments = InitArguments());

       :param arguments: ``struct`` object with valid initialization arguments

       .. warning:: Valid until 3.7

    .. cpp:function:: template <class... Args> ScopeGuard(Args&&... args);

        :param args: arguments to pass to `Kokkos::initialize <initialize.html#kokkosinitialize>`_

	Possible implementation:

	.. code-block:: cpp

	   template <class... Args> ScopeGuard(Args&&... args){ initialize(std::forward<Args>(args)...); }

    .. cpp:function:: ~ScopeGuard();

       Destructor

       Possible implementation:

       .. code-block:: cpp

	  ~ScopeGuard() { finalize(); }

    .. cpp:function:: ScopeGuard(ScopeGuard const&) = delete;

       Copy constructor

    .. cpp:function:: ScopeGuard(ScopeGuard&&) = delete;

       Move constructor

    .. cpp:function:: ScopeGuard& operator=(ScopeGuard const&) = delete;

       Copy assignment operator

    .. cpp:function:: ScopeGuard& operator=(ScopeGuard&&) = delete;

       Move assignment operator

Notes
-----

- In the constructors, all the parameters are passed to the ``Kokkos::initialize`` called internally.
  See `Kokkos::initialize <initialize.html#kokkosinitialize>`_ for more details.


- Since Kokkos version 3.7, ``ScopeGuard`` unconditionally forwards the provided
  arguments to `Kokkos::initialize <initialize.html#kokkosinitialize>`_, which means they have the same
  preconditions.  Until version 3.7, ``ScopeGuard`` was calling
  ``Kokkos::initialize`` in its constructor only if ``Kokkos::is_initialized()`` was
  ``false``, and it was calling ``Kokkos::finalize`` in its destructor only if it
  called ``Kokkos::initialize`` in its constructor.

  We dropped support for the old behavior.  If you think you really need it, you may do:

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

`Kokkos::initialize <initialize.html#kokkosinitialize>`_, `Kokkos::finalize <finalize.html#kokkosfinalize>`_
