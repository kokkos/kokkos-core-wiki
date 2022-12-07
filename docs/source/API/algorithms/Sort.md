
# Sort

template< class DstViewType , class SrcViewType >
  struct copy_functor { }

template< class DstViewType, class PermuteViewType, class SrcViewType>
  struct copy_permute_functor { }

class BinSort {

* template< class DstViewType , class SrcViewType >  struct copy_functor { }

* template< class DstViewType, class PermuteViewType, class SrcViewType >  struct copy_permute_functor { }


*   template<class ValuesViewType>  void sort( ValuesViewType const & values, int values_range_begin, int values_range_end) const  { }

*   template<class ValuesViewType>  void sort( ValuesViewType const & values ) const  { }

}


template<class KeyViewType>  struct BinOp1D  { }

template<class KeyViewType>  struct BinOp3D  { }


template<class ViewType>  void sort( ViewType const & view )  { }

template<class ViewType>  void sort( ViewType view, size_t const begin, size_t const end )  {  }

# Sorting with nested policies (team- and thread-level)

Parallel sort functions for use within ``TeamPolicy`` kernels. These perform sorting using team-level (``TeamThreadRange``) or thread-level (``ThreadVectorRange``) parallelism.

Header: Kokkos_NestedSort.hpp

## Synopsis
```
//namespace Kokkos::Experimental

template <class TeamMember, class ViewType>
KOKKOS_INLINE_FUNCTION void sort_team(const TeamMember& t, const ViewType& view);

template <class TeamMember, class ViewType, class Comparator>
KOKKOS_INLINE_FUNCTION void sort_team(const TeamMember& t, const ViewType& view, const Comparator& comp);

template <class TeamMember, class KeyViewType, class ValueViewType>
KOKKOS_INLINE_FUNCTION void sort_by_key_team(
  const TeamMember& t, const KeyViewType& keyView, const ValueViewType& valueView);

template <class TeamMember, class KeyViewType, class ValueViewType, class Comparator>
KOKKOS_INLINE_FUNCTION void sort_by_key_team(
  const TeamMember& t, const KeyViewType& keyView, const ValueViewType& valueView, const Comparator& comp);

template <class TeamMember, class ViewType>
KOKKOS_INLINE_FUNCTION void sort_thread(const TeamMember& t, const ViewType& view);

template <class TeamMember, class ViewType, class Comparator>
KOKKOS_INLINE_FUNCTION void sort_thread(const TeamMember& t, const ViewType& view);

template <class TeamMember, class ViewType, class Comparator>
KOKKOS_INLINE_FUNCTION void sort_thread(const TeamMember& t, const ViewType& view, const Comparator& comp);

template <class TeamMember, class KeyViewType, class ValueViewType>
KOKKOS_INLINE_FUNCTION void sort_by_key_thread(
  const TeamMember& t, const KeyViewType& keyView, const ValueViewType& valueView);

template <class TeamMember, class KeyViewType, class ValueViewType, class Comparator>
KOKKOS_INLINE_FUNCTION void sort_by_key_thread(
  const TeamMember& t, const KeyViewType& keyView, const ValueViewType& valueView, const Comparator& comp);
```

``sort_team`` and ``sort_by_key_team`` internally use the entire team, so they may be called inside the top level of ``TeamPolicy`` lambdas and functors. ``sort_thread`` and ``sort_by_key_thread`` use the vector lanes of a thread, so they may be called either within ``TeamPolicy`` or ``TeamThreadRange`` loops.

The ``sort_by_key`` functions sort ``keyView``, while simultaneously applying the same permutation to the elements of ``valueView``. It is equivalent to sorting ``(key[i], value[i])`` tuples according by key. An example of where this is commonly used is to sort the entries and values within each row of a CRS (compressed row sparse) matrix.

Versions taking a ``Comparator`` object will use it to order the keys. ``Comparator::operator()`` should be a const member function that accept two keys ``a`` and ``b``, and returns a bool that is true if and only if ``a`` goes before ``b`` in the sorted list. For versions not taking a ``Comparator`` object, keys are sorted into ascending order (according to ``operator<``). For example, this comparator will sort a list of ``int`` in ascending order:
```
struct IntComparator {
  KOKKOS_FUNCTION constexpr bool operator()(const int& a, const int& b) const {
    return a < b;
  }
};
```

## Additional Information

- All functions include a final barrier at their level of parallelism, so all elements of ``view``/``keyView``/valueView`` may be accessed immediately after they return.

- These functions can operate on views in both global and scratch memory spaces.

- These functions use the bitonic sorting algorithm, which is not stable. This means if a key is repeated in the input, then the values corresponding to that key might be in any order after doing a sort by key.

## Example

```
#include <Kokkos_Core.hpp>
#include <Kokkos_NestedSort.hpp>
#include <Kokkos_Random.hpp>

int main(int argc, char* argv[]) {
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using TeamPol = Kokkos::TeamPolicy<ExecSpace>;
  using TeamMem = typename TeamPol::member_type;
  Kokkos::initialize(argc, argv);
  {
    int n = 10;
    Kokkos::Random_XorShift64_Pool<ExecSpace> rand_pool(13718);
    Kokkos::View<int**, ExecSpace> A("A", n, n);
    Kokkos::fill_random(A, rand_pool, 100);
    Kokkos::parallel_for(
        TeamPol(n, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const TeamMem& t)
        {
          //Sort a row of A using the whole team.
          auto A_row_i = Kokkos::subview(A, t.league_rank(), Kokkos::ALL());
          Kokkos::Experimental::sort_team(t, A_row_i);
        });
    auto Ahost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
    std::cout << "A, sorted by row:\n";
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        std::cout << Ahost(i, j) << ' ';
      }
      std::cout << '\n';
    }
    int vectorLen = TeamPol::vector_length_max();
    Kokkos::parallel_for(
        TeamPol(1, Kokkos::AUTO(), vectorLen),
        KOKKOS_LAMBDA(const TeamMem& t)
        {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(t, n),
            [=](int i)
            {
              //Now sort a column of A by using just this thread.
              auto A_col_i = Kokkos::subview(A, Kokkos::ALL(), i);
              Kokkos::Experimental::sort_thread(t, A_col_i);
            });
        });
    Kokkos::deep_copy(Ahost, A);
    std::cout << "\nA, now sorted by col:\n";
    for(int i = 0; i < n; i++) {
      for(int j = 0; j < n; j++) {
        std::cout << Ahost(i, j) << ' ';
      }
      std::cout << '\n';
    }
  }
  Kokkos::finalize();
  return 0;
}
```

