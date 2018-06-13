# `Kokkos::Timer`

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
  Kokkos::Timer timer;
  double time = timer.seconds();
  timer.reset();
  ```

Timer is a simple construct to measure time. 

## Synopsis 
  ```c++
  class Timer {
    typedef Timer execution_policy;

    //Constructors
    Timer();
    Timer(Timer&&) = default;
    Timer(const Timer&) = default;
    ~Timer() = default;

    //Member functions
    double seconds(); 
    void reset();
  };
  ```

## Public Class Members

### Constructors
 
 * `Timer()`: Default Constructor. Sets the start time. 
 * `void reset()`: Resets the start time. 
 * `double seconds()`: Returns the number of seconds since last call to `reset()` or construction. 
  
## Examples

  ```c++
    Timer timer;
    // ...
    double time1 = timer.seconds();
    timer.reset();
    // ...
    double time2 = timer.seonds();
  ```

