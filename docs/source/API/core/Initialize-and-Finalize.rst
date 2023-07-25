Initialize and Finalize
=======================

Kokkos::initialize
------------------

Initializes Kokkos internal objects and all enabled Kokkos backends.

See `Kokkos::initialize <initialize_finalize/initialize.html>`_ for details.


Kokkos::finalize
----------------

Shutdown Kokkos initialized execution spaces and release internally managed resources.

See `Kokkos::finalize <initialize_finalize/finalize.html>`_ for details.


Kokkos::ScopeGuard
------------------

``Kokkos::ScopeGuard`` is a class which aggregates the resources managed by Kokkos. ScopeGuard will call ``Kokkos::initialize`` when constructed and ``Kokkos::finalize`` when destructed, thus the Kokkos context is automatically managed via the scope of the ScopeGuard object.

See `Kokkos::ScopeGuard <initialize_finalize/ScopeGuard.html>`_ for details.

ScopeGuard aids in the following common mistake which is allowing Kokkos objects to live past ``Kokkos::finalize``:

.. code-block:: cpp

  int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    Kokkos::View<double*> my_view("my_view", 10);
    Kokkos::finalize();
    // my_view destructor called after Kokkos::finalize !
  }


.. raw:: html

  <iframe width="800px" height="300px" src="https://godbolt.org/e?hideEditorToolbars=true#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:1,endLineNumber:9,positionColumn:1,positionLineNumber:9,selectionStartColumn:1,selectionStartLineNumber:9,startColumn:1,startLineNumber:9),source:'%23include+%3CKokkos_Core.hpp%3E%0A%0Aint+main(int+argc,+char**+argv)+%7B%0A++Kokkos::initialize(argc,+argv)%3B%0A++Kokkos::View%3Cdouble*%3E+my_view(%22my_view%22,+10)%3B%0A++Kokkos::finalize()%3B%0A++//+my_view+destructor+called+after+Kokkos::finalize+!!%0A%7D%0A'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',m:100,n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:clang1600,deviceViewOpen:'1',filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!((name:kokkos,ver:'4100')),options:'',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+x86-64+clang+16.0.0+(Editor+%231)',t:'0')),header:(),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4"></iframe>

Switching to ``Kokkos::ScopeGuard`` fixes it:

.. code-block:: cpp

  int main(int argc, char** argv) {
    Kokkos::ScopeGuard kokkos(argc, argv);
    Kokkos::View<double*> my_view("my_view", 10);
    // my_view destructor called before Kokkos::finalize
    // ScopeGuard destructor called, calls Kokkos::finalize
  }

In the above example, ``my_view`` will not go out of scope until the end of the main() function.  Without ``ScopeGuard``, ``Kokkos::finalize`` will be called before ``my_view`` is out of scope.  With ``ScopeGuard``, ``ScopeGuard`` will be dereferenced (subsequently calling ``Kokkos::finalize``) after ``my_view`` is dereferenced, which ensures the proper order during shutdown.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./initialize_finalize/initialize
   ./initialize_finalize/finalize
   ./initialize_finalize/ScopeGuard
   ./initialize_finalize/InitializationSettings
   ./initialize_finalize/InitArguments
