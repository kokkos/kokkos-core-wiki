Kokkos provides Reducers for the most common reduction types:

* [Sum](Kokkos%3A%3ASum): For simple Summations
* [Prod](Kokkos%3A%3AProd): Computing the product of all input values
* Min: Finding the minimum value
* Max: Finding the maximum value
* MinMax: Finding the minimum and the maximum value
* MinLoc: Retrieve the minimum value as well as its associated index
* MaxLoc: Retrieve the maximum value as well as its associated index
* MinMaxLoc: Find both the maximum and minimum value as well as their associated indices
* BAnd: Do a binary “and” reduction
* BOr: Do a binary “or” reduction
* LAnd: Do a logical “and” reduction
* LOr: Do a logical “or” reduction

These reducers work only for scalar data, i.e. you can’t have a runtime length array as the reduction type (for example finding the minimum values for each vector in a multi vector concurrently).
Generally the Reducers are templated on the Scalar type for the reduction as well as an optional template parameter for the memory space of the result (more on that later). The `MinLoc`, `MaxLoc` and `MinMaxLoc` reducers are additionally templated on the index type. 

The following is an example for doing a simple min-reduction, finding the minimal value in a discretization of a parable.
```c++
double min;

Kokkos::parallel_reduce( “MinReduce”, N, KOKKOS_LAMBDA (const int& x, double& lmin) {
  double val = (1.0*x- 7.2) * (1.0*x- 7.2) + 3.5;
  if( val < lmin ) lmin = val; 
}, Kokkos::Min<double>(min));

printf(“Min: %lf\n”, min);
```
In this example the `Min` reducer was templated on the reducing type `double` and the variable to store the result was taken in by reference. Note that the reducer is only used to combine values from different threads. The per thread reduction is still performed explicitly. One could have used the reducer for that as well through a reducer instance:

```c++
double min;

Kokkos::Min<double> min_reducer(min);
Kokkos::parallel_reduce( “MinReduce”, N, KOKKOS_LAMBDA (const int& x, double& lmin) {
  double val = (1.0*x- 7.2) * (1.0*x- 7.2) + 3.5;
  min_reducer.join(lmin, val); 
}, min_reducer);

printf(“Min: %lf\n”, min);
```

For the `MinLoc`, `MaxLoc` and `MinMaxLoc` reducers the reduction type is actually different than the scalar type. Those types are accessible through a `value_type` typedef. 
`MinLoc` and `MaxLoc` have value types which have a simple `val` and `loc` member to store the reduction value and the index respectively. Note that index can be a struct itself, for example to store a multi dimensional index result (see later). 

```c++
typedef Kokkos::MinLoc<double,int>::value_type minloc_type;
minloc_type minloc;

Kokkos::parallel_reduce( “MinLocReduce”, N, KOKKOS_LAMBDA (const int& x, minloc_type& lminloc) {
  double val = (1.0*x- 7.2) * (1.0*x- 7.2) + 3.5;
  if( val < lminloc.val ) { lminloc.val = val; lminloc.loc = i; }
}, Kokkos::MinLoc<double,int> minloc_reducer(minloc));

printf(“Min: %lf at %i\n”, minloc.val, minloc.loc);
```

Reducers can be used in nested reductions. This example also makes use of a 2D index type to find the minimum and maximum value of a matrix as well as their indices. 

```c++
Kokkos::View<double**> A(“A”,N,M);
// fill A

// Create a variable for the result
typedef Kokkos::MinMaxLoc<double, Kokkos::pair<int,int>> reducer_type;
typedef reducer_type::value_type value_type;
value_type minmaxloc

typedef Kokkos::TeamPolicy<>::member_type team_type;

// Start a team parallel reduce
Kokkos::parallel_reduce( “MinLocReduce”, Kokkos::TeamPolicy<>(N,AUTO), 
    KOKKOS_LAMBDA (const team_type& team, value_type& team_minmaxloc) {

  // Create a temporary to store the reduction value for the row
  value_type row_minmaxloc;
  int n = team.league_rank();

  // Run a nested parallel reduce with the team over the row
  Kokkos::parallel_reduce( Kokkos::TeamThreadRange(team, M), 
      [=] (const int& m, value_type& thread_minmaxloc) {
    double val = A(n,m);
    
    // Check whether this is a new minimum or maximum value
    if(val < thread_minmaxloc.min_val) {
      thread_minmaxloc.min_val = val;
      thread_minmaxloc.min_loc = Kokkos::pair<int,int>(n,m);
    }
    if(val > thread_minmaxloc.max_val) {
      thread_minmaxloc.max_val = val;
      thread_minmaxloc.max_loc = Kokkos::pair<int,int>(n,m);
    }

  }, reducer_type(row_minmaxloc));
  
  // One guy in the team should contribute to the whole
  // Note: for a min or max reduction it wouldn't hurt if 
  //       every team member did this
  Kokkos::single(Kokkos::PerTeam(team), [=] () {
    if( row_minmaxloc.min_val < team_minmax.min_val ) {
      team_minmax.min_val = row_minmax.min_val;
      team_minmax.min_loc = row_minmax.min_loc;
    }
    if( row_minmaxloc.max_val < team_minmax.max_val ) {
      team_minmax.max_val = row_minmax.max_val;
      team_minmax.max_loc = row_minmax.max_loc;
    }
  }
}, reducer_type(minmaxloc));

printf(“Min %lf at (%i, %i)\n”,minmaxloc.min_val, minmaxloc.min_loc.first, minmaxloc.min_loc.second);
printf(“Min %lf at (%i, %i)\n”,minmaxloc.max_val, minmaxloc.max_loc.first, minmaxloc.max_loc.second);
```