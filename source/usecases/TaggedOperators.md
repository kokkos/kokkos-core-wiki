# Tagged Operators

## Dealing with pre-C++17

Many applications and libraries in HPC are written using an Object Oriented design.
This has numerous advantages in terms of code organization and potentially reuse.
Unfortunately it comes with a significant draw back as far as using Kokkos is concerned,
if C++17 is not an option.

Consider an application for simulating the time evolution of a particle system.
In an Object Oriented design, it would likely contain something like a class `ParticleInteractions`,
which holds references to data structures for particles and interaction potentials.
It would also contain some member function `compute`, which would loop over the particles
and calculate the interactions:

```c++
class ParticleInteractions {
  ParticlePositions pos;
  ParticleForces forces;
  ParticleNeighbors neighbors;
  PairForce force;

public:
  void compute() {
    for(int i=0; i<pos.extent(0); i++) {
      for(int j=0; j<neighbors.count(i); j++) {
        forces(i) = force(pos(i),pos(neighbors(i,j)));
      }
    }
  }
};
```

A simple way of parallelising this is to add a `parallel_for` inside the member function `compute`.

```c++
class ParticleInteractions {
  ...
  void compute() {
    parallel_for("Compute", KOKKOS_LAMBDA (const int i) {
      for(int j=0; j<neighbors.count(i); j++) {
        forces(i) = force(pos(i),pos(neighbors(i,j)));
      }
    });
  }
};
```

And indeed on non-accelerator based systems this would work. But on systems where the `parallel_for`
dispatches work to an accelerator, this approach would likely fail with an access error.

The reason for that is, that lambdas inside of a class member function do not capture other
class members individually, they capture the entire class as a whole.
More precisely, in pre-C++17 they capture the `this` pointer at the scope of `compute`.

In effect it is as if one would have written:
```c++
class ParticleInteractions {
  ...
  void compute() {
    parallel_for("Compute", KOKKOS_LAMBDA (const int i) {
      for(int j=0; j<this->neighbors.count(i); j++) {
        this->forces(i) = this->force(this->pos(i),this->pos(this->neighbors(i,j)));
      }
    });
  }
};
```
If `this` is not dereferencable in the scope of the execution space, the execution will fail.

In C++17 the situation can be rectified by using the `[*this]` capture clause. In that case,
the whole class instance will be captured and copied to the accelerator as part of the dispatch.

One way to deal with this situation pre-C++17 is to simply make a corresponding operator for the
compute function, and dispatch the class itself as a functor:
```c++
class ParticleInteractions {
  ...
  void compute() {
    parallel_for("Compute", *this);
  }
  KOKKOS_FUNCTION
  void operator() (const int i) const {
    for(int j=0; j<neighbors.count(i); j++) {
      forces(i) = force(pos(i),pos(neighbors(i,j)));
    }
  }
};
```

In fact this may even have clarity enhancing qualities, since it separates the work items of the parallel operation,
from the dispatch itself and its associated pre and post launch work.

But what if you have more than one parallel operation in a class?
This is where Kokkos's tagged dispatch comes into play.
In order for a class to have multiple operators, Kokkos allows these operators to have an unsued additional parameter
which is used in overload resolution.
This parameter is given as an additional template parameter to the execution policy during dispatch.
A good practice is to create `Tag-Classes` as nested definitions inside the original object:

```c++
class ParticleInteractions {
  class TagPhase1 {};
  class TagPhase2 {};
  ...

  void compute() {
    parallel_for("Compute1", RangePolicy<TagPhase1>(0,N), *this);
    parallel_for("Compute2", RangePolicy<TagPhase2>(0,N), *this);
  }
  KOKKOS_FUNCTION
  void operator() (TagPhase1, const int i) const {
    ...
  }
  KOKKOS_FUNCTION
  void operator() (TagPhase2, const int i) const {
    ...
  }
};
```

## Templating Operators

Another useful application of the tagged interface is the opportunity to template operators.
A particular use case has been the conversion of runtime loop parameters into compile time ones:
