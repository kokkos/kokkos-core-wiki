
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


template<class ViewType>  void sort( ViewType const & view , bool const always_use_kokkos_sort = false)  { }

template<class ViewType>  void sort( ViewType view, size_t const begin, size_t const end )  {  }
