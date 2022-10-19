# MPI Halo Exchange

Kokkos and MPI are complementary programming models: Kokkos is designed to handle
parallel programming within a shared-memory space, and MPI is designed to handle parallel programming
between multiple distributed memory spaces.
In order to create a fully scalable parallel program, it is often necessary to use both
Kokkos and MPI.
This Use Case document walks through an example of how MPI and Kokkos can work together.

## Sending a single message

MPI is based around message-passing semantics, and one of the simplest operations in MPI is sending
a single message.
Typically, this message is contained in a single contiguous memory allocation, and consists of some
number of values of the same type (for example, double-precision floating-point values).
Notice that this definition of a message is very similar to the definition of a `Kokkos::View`:
a collection of values of the same type, which is often contiguous.
As such, it is often straightforward to send the contents of a `Kokkos::View` as a single MPI message.
The way to do this is to obtain what MPI needs: a pointer to the start of the message allocation via `Kokkos::View::data()`
and the number of items in the message via `Kokkos::View::size()`.
Here is an example that sends `double` values from one rank to another.

```c++
int source_rank = 1;
int destination_rank = 1;
int number_of_doubles = 12;
int tag = 42;
Kokkos::View<double*> buffer("buffer", number_of_doubles);
int my_rank;
MPI_Comm comm = MPI_COMM_WORLD;
MPI_Comm_rank(comm, &my_rank);
if (my_rank == source_rank) {
  MPI_Send(buffer.data(), int(buffer.size()), MPI_DOUBLE, destination_rank, tag, comm);
} else if (my_rank == destination_rank) {
  MPI_Recv(buffer.data(), int(buffer.size()), MPI_DOUBLE, destination_rank, tag, comm);
}
```

## CUDA-Aware MPI

One common concern for programmers who are using CUDA GPU parallelism through Kokkos as well as MPI is
how to use MPI to communicate between two ranks which are each using CUDA parallelism.
The current path forward is quite simple: just pass your allocation pointers as before and it should
"just work".
In particular, the MPI libraries installed on GPU clusters should be compiled with what we call
CUDA-aware support.
This means that the MPI library is aware of CUDA and will call CUDA functions which determine whether
the user's allocation pointer points to device memory, host memory, or managed (UVM) memory.
If the pointer points to device or managed memory, the MPI library can optimize the communication by
calling CUDA functions to copy the relevant memory from one place to another on the same GPU,
or from one GPU to another through PCIe or NVIDIA NVLINK if available.
As such, the example above continues to work even if `Kokkos::DefaultExecutionSpace` is `Kokkos::Cuda`.

## Separating out messages

There is often a need in unstructured MPI-based codes to determine what subsets of data need to be
packed into which messages and sent to which other ranks, based on less structured information.
For example, assume we have a simulation composed of thousands of "elements", and each element is
owned by one MPI rank.
Other ranks may have redundant copies of that element, but some fundamental decision-making related
to that element must be done by the MPI rank that owns it.
Suppose further that on each MPI rank there exists an array that maps each element, whether owned
or redudantly copied, to the (possibly different) MPI rank which owns that element.
We can filter out the subset of these elements that are associated with a given owner using
[`Kokkos::parallel_scan`](../API/core/parallel-dispatch/parallel_scan) and subsequently pack messages using [`Kokkos::parallel_for`](../API/core/parallel-dispatch/parallel_for).

## Identifying subset indices

For the filter-out step, we simply need to identify which ranks (keys) are the same as some
known destination, and if they are then we number them consecutively in the order they appear
(which is a scan operation).
Here is an example which filters out a subset of items:

```c++
// an exclusive scan functor, which during the final pass will
// assign into the compressed subset indices
class subset_scanner {
public:
  using execution_space = Kokkos::DefaultExecutionSpace;
  using value_type = int;
  using size_type = int;
  subset_scanner(
      Kokkos::View<int*, execution_space> keys_in,
      int desired_key_in,
      Kokkos::View<int*, execution_space> subset_indices_in)
    :m_keys(keys_in)
    ,m_desired_key(desired_key_in)
    ,m_subset_indices(subset_indices_in)
  {}
  KOKKOS_INLINE_FUNCTION void operator()(int i, int& update, const bool final_pass) const {
    bool is_in = (m_keys[i] == m_desired_key);
    if (final_pass && is_in) {
      m_subset_indices[update] = i;
    }
    update += (is_in ? 1 : 0);
  }
  KOKKOS_INLINE_FUNCTION void init(int& update) const {
    update = 0;
  }
  KOKKOS_INLINE_FUNCTION void join(int& update, const int& input) const {
    update += input;
  }
private:
  Kokkos::View<int*, execution_space> m_keys;
  int m_desired_key;
  Kokkos::View<int*, execution_space> m_subset_indices;
};

Kokkos::View<int*> find_subset(Kokkos::View<int*> keys, int desired_key) {
  int subset_size = 0;
  Kokkos::parallel_reduce(keys.size(), KOKKOS_LAMBDA(int i, int& local_sum) {
    return keys[i] == desired_key ? 1 : 0;
  }, subset_size);
  Kokkos::View<int*> subset_indices("subset indices", subset_size);
  Kokkos::parallel_scan(keys.size(), subset_scanner(keys, desired_key, subset_indices));
  return subset_indices;
}
```

## Extracting subset message

Once we are able to produce a list of subset indices (those indices of elements which will be transmitted in one message),
we can use that list of indices to extract a subset of the simulation data to send.
Here, let us assume that we have a [`Kokkos::View`](../API/core/view/view) which stores one floating-point value per element, and we want
to extract a message containing only the floating-point values for the relevant subset.

```c++
Kokkos::View<double*> pack_message(Kokkos::View<double*> all_element_data, Kokkos::View<int*> subset_indices) {
  Kokkos::View<double*> message("message", subset_indices.size());
  Kokkos::parallel_for(subset_indices.size(), KOKKOS_LAMBDA(int i) {
    message[i] = all_element_data[subset_indices[i]];
  });
  return message;
}
```
