## ScatterView averaging elements to nodes

In order to demonstrate a typical use case for `Kokkos::ScatterView`, we can think of
a finite element program where the only information available is a mapping from elements
to nodes, and we would like to average some quantity from the elements to the nodes.
This average is the ratio of two sums, namely the sum of the adjacent element quantities
divided by the sum of adjacent elements.

### Computing the number of adjacent elements

Even just computing the number of elements adjacent to a node will already demonstrate most
of the necessary workflow around `Kokkos::ScatterView`.
The algorithm is as follows: we will iterate over mesh elements, in parallel, and each mesh
element will identify its nodes and add one to an array entry specific to that node.
Those entries are ultimately stored in a `Kokkos::View` with one entry per node, but
during the algorithm they will be accessed through a `Kokkos::ScatterView` in order to
prevent data races.

```cpp
Kokkos::View<int*> count_adjacent_elements(Kokkos::View<int**> elements_to_nodes, int number_of_nodes) {
  Kokkos::View<int*> elements_per_node("elements_per_node", number_of_nodes);
  auto scatter_elements_per_node = Kokkos::Experimental::create_scatter_view(elements_per_node);
  Kokkos::parallel_for(elements_to_nodes.extent(0), KOKKOS_LAMBDA(int element) {
    auto access_elements_per_node = scatter_elements_per_node.access();
    for (int node_of_element = 0; node_of_element < elements_to_nodes.extent(1); ++node_of_element) {
      int node = elements_to_nodes(element, node_of_element);
      access_elements_per_node(node) += 1;
    }
  });
  Kokkos::Experimental::contribute(elements_per_node, scatter_elements_per_node);
  return elements_per_node;
}
```

### Computing the value sums at nodes

Computing the sum of the element values adjacent to a node is almost identical to computing
the number of elements around a node:

```cpp
Kokkos::View<double*> sum_to_nodes(Kokkos::View<int**> elements_to_nodes, int number_of_nodes,
    Kokkos::View<double*> element_values) {
  Kokkos::View<double*> node_values("node_values", number_of_nodes);
  auto scatter_node_values = Kokkos::Experimental::create_scatter_view(node_values);
  Kokkos::parallel_for(elements_to_nodes.extent(0), KOKKOS_LAMBDA(int element) {
    auto access_node_values = scatter_node_values.access();
    for (int node_of_element = 0; node_of_element < elements_to_nodes.extent(1); ++node_of_element) {
      int node = elements_to_nodes(element, node_of_element);
      access_node_values(node) += element_values(element);
    }
  });
  Kokkos::Experimental::contribute(node_values, scatter_node_values);
  return node_values;
}
```

### Computing the full average

Now that we have two sums at each node, it is sufficient to use one final loop over nodes
to take the ratio of these two sums and define the average.
This function will be structured by assuming that the number of elements adjacent to each
node has been pre-computed.

```cpp
Kokkos::View<double*> average_to_nodes(Kokkos::View<int**> elements_to_nodes, int number_of_nodes,
    Kokkos::View<double*> element_values,
    Kokkos::View<int*> elements_per_node) {
  auto node_values = sum_to_nodes(elements_to_nodes, number_of_nodes, element_values);
  Kokkos::parallel_for(number_of_nodes, KOKKOS_LAMBDA(int node) {
    node_values[node] /= elements_per_node[node];
  });
  return node_values;
}
```
