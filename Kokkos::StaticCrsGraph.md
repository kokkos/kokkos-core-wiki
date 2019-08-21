# Kokkos::StaticCrsGraph

The StaticCrsGraph is a Compressed row storage array with the row map, the column indices and the non-zero entries stored in 3 different Kokkos::Views.  Appropriate types and functions are provided to simplify manipulation and access to CRS data on either a host or device.

