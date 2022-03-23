# Bitset

Kokkos's ```Bitset``` represents a thread safe view to a fixed-size (at run-time) sequence of N bits.

Kokkos's ```ConstBitset``` represents a thread safe view to a read-only fixed-size (at run-time) sequence of N bits.

Header file: `Kokkos_Bitset.hpp`

## Kokkos::Bitset API

```c++
template <typename Device>
class Bitset {
 public:
  using execution_space = typename Device::execution_space;
  using size_type       = unsigned;

  // Host/Device: Construct a bitset with arg_size bits
  Bitset(unsigned arg_size = 0u);

  // Host/Device: return the number of bits
  unsigned size() const;

  // Host: return the number of bits which are set to 1
  unsigned count() const;

  // Host: set all bits to 1
  void set();

  // Host/Device: set all bits to 0
  void reset();
  void clear();

  // Device: set i'th bit to 1
  void set(unsigned i);

  // Device: set i'th bit to 0
  void reset(unsigned i);

  // Device: return true if and only if the i'th bit set to 1
  bool test(unsigned i) const;

  // Host/Device: used with find_any_*set_near functions
  // returns the max number of times those functions should be call
  // when searching for an available bit
  unsigned max_hint() const;

  // Host/Device: used with find_any_*set_near functions
  // to indicate the scan_direction
  // 
  // BIT_SCAN_FORWARD_* scans for the bit in the foward direction
  // BIT_SCAN_REVERSE_* scans for the bit in the foward direction
  //
  // If the bit was not found:
  //    *_MOVE_HINT_FORWARD selects a new hint past the current hint
  //    *_MOVE_HINT_BACKWARD selects a new hint before the current hint
  static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_FORWARD;
  static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_FORWARD;
  static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_BACKWARD;
  static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_BACKWARD;

  // Host/Device: find a bit set to 1 near the hint
  // returns a pair<bool, unsigned> where if result.first is true then
  // result.second is the bit found and if result.first is false the
  // result.second is a new hint
  Kokkos::pair<bool, unsigned> find_any_set_near(
      unsigned hint,
      unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const;

  // Host/Device: find a bit set to 0 near the hint
  // returns a pair<bool, unsigned> where if result.first is true then
  // result.second is the bit found and if result.first is false the
  // result.second is a new hint
  Kokkos::pair<bool, unsigned> find_any_unset_near(
      unsigned hint,
      unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const;

  // Host/Device: the bits are allocated on the device
  constexpr bool is_allocated() const;
};
```

## ConstBitset API

```c++
template <typename Device>
class ConstBitset {
 public:
  using execution_space = typename Device::execution_space;
  using size_type       = unsigned;

  // Host/Device: Construct a bitset with no bits
  ConstBitset();

  // Host/Device: Copy construct a Bitset to a ConstBitset
  ConstBitset(Bitset<Device> const& rhs);

  // Host/Device: Copy assign a Bitset to a ConstBitset
  ConstBitset& operator=(Bitset<Device> const& rhs);

  ConstBitset(ConstBitset const& rhs) = default;
  ConstBitset& operator=(ConstBitset const& rhs) = default;

  // Host/Device: return the number of bits
  unsigned size() const;

  // Host/Device: return the number of bits which are set to 1
  unsigned count() const;

  // Host/Device: Return true if and only if the i'th bit set to 1
  bool test(unsigned i) const;
};
```

## Bitset/ConstBitset deep_copy free functions API:

```c++
template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src);

template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src);

template <typename DstDevice, typename SrcDevice>
void deep_copy(ConstBitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src);

```

