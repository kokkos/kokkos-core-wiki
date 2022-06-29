

## Synopsis

```c++
  // This is not an actual class, it just describes the concept in shorthand
  class ExecutionSpaceConcept {
    public: 
      typedef ExecutionSpaceConcept execution_space;
      typedef ... memory_space;
      typedef Device<execution_space, memory_space> device_type;
      typedef ... scratch_memory_space;
      typedef ... array_layout;
      

      ExecutionSpaceConcept();
      ExecutionSpaceConcept(const ExecutionSpaceConcept& src);

      const char* name() const;
      void print_configuration(std::ostream ostr&) const;
      void print_configuration(std::ostream ostr&, bool details) const;
      
      bool in_parallel() const;
      int concurrency() const;

      void fence() const;
  };

  template<class MS>
  struct is_execution_space {
    enum { value = false };
  };

  template<>
  struct is_execution_space<ExecutionSpaceConcept> {
    enum { value = true };
  };
```


## Public Class Members

### Typedefs

  * `execution_space`: The self type;
  * `memory_space`: The default [`MemorySpace`](MemorySpaceConcept) to use when executing with `ExecutionSpaceConcept`.  
                    Kokkos guarantees that `Kokkos::SpaceAccessibility<Ex, Ex::memory_space>::accessible` will be `true` 
                    (see [`Kokkos::SpaceAccessibility`](Kokkos%3A%3ASpaceAccessibility))
  * `device_type`: `DeviceType<execution_space,memory_space>`.
  * `array_layout`: The default [`ArrayLayout`](ArrayLayoutConcept) recommended for use with `View` types accessed from `ExecutionSpaceConcept`.
  * `scratch_memory_space`: The [`ScratchMemorySpace`](ScratchMemorySpaceConcept) that parallel patterns will use for allocation of scratch memory 
                            (for instance, as requested by a [`Kokkos::TeamPolicy`](Kokkos%3A%3ATeamPolicy))

### Constructors

  * `ExecutionSpaceConcept()`: Default constructor.
  * `ExecutionSpaceConcept(const ExecutionSpaceConcept& src)`: Copy constructor.

### Functions

  * `const char* name() const;`: *Returns* the label of the execution space instance.
  * `bool in_parallel() const;`: *Returns* a value convertible to `bool` indicating whether or not the caller is executing as part of a Kokkos parallel pattern.
        *Note:* as currently implemented, there is no guarantee that `true` means the caller is necessarily executing as 
        part of a pattern on the particular instance `ExecutionSpaceConcept`; just *some* instance of `ExecutionSpaceConcept`.  This may be strengthened in the future.
  * `int concurrency() const;` *Returns* the maximum amount of concurrently executing work items in a parallel setting, i.e. the maximum number of threads utilized by an execution space instance.
  * `void fence() const;` *Effects:* Upon return, all parallel patterns executed on the instance `ExecutionSpaceConcept` are guaranteed to have completed, 
                          and their effects are guaranteed visible to the calling thread. 
                          *Note:* This *cannot* be called from within a parallel pattern.  Doing so will lead to unspecified effects 
                          (i.e., it might work, but only for some execution spaces, so be extra careful not to do it).
  * `void print_configuration(std::ostream ostr) const;`: *Effects:* Outputs the configuration of `ex` to the given `std::ostream`.
        *Note:* This *cannot* be called from within a parallel pattern.

## Non Member Facilities

  * `template<class MS> struct is_execution_space;`: typetrait to check whether a class is a execution space.
  * `template<class S1, class S2> struct SpaceAccessibility;`: typetraits to check whether two spaces are compatible (assignable, deep_copy-able, accessable). 
          (see [`Kokkos::SpaceAccessibility`](Kokkos%3A%3ASpaceAccessibility))

