
# `Bitset`

Header file: `Kokkos_Bitset.hpp`

Usage:

```Kokkos::Bitset``` represents a thread safe view to a fixed-size (at run-time) sequence of N bits.

```Kokkos::ConstBitset``` represents a thread safe view to a read-only fixed-size (at run-time) sequence of N bits.

## Interface

```c++
template <typename Device>
class Bitset;
```

### Parameters

   * ```Device``` : Device that physically contains the bits.

## Public Class Members

### Static Constants

```c++
  static constexpr unsigned BIT_SCAN_REVERSE   = 1u
  static constexpr unsigned MOVE_HINT_BACKWARD = 2u

  static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_FORWARD = 0u
  static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_FORWARD = BIT_SCAN_REVERSE
  static constexpr unsigned BIT_SCAN_FORWARD_MOVE_HINT_BACKWARD = MOVE_HINT_BACKWARD
  static constexpr unsigned BIT_SCAN_REVERSE_MOVE_HINT_BACKWARD = BIT_SCAN_REVERSE | MOVE_HINT_BACKWARD
```

* ```BIT_SCAN_REVERSE``` : Bit mask for scanning direction
* ```MOVE_HINT_BACKWARD``` : Bit mask for hint direction

* ```BIT_SCAN_FORWARD_MOVE_HINT_FORWARD``` : When passed as ```scan_direction``` to
                                             ```find_any_set_near(...)``` or ```find_any_reset_near(...)```,
                                             scans for the bit in the forward (increasing index) direction.
                                             If the bit was not found, selects a new hint past the current hint.
* ```BIT_SCAN_REVERSE_MOVE_HINT_FORWARD``` : When passed as ```scan_direction``` to
                                             ```find_any_set_near(...)``` or ```find_any_reset_near(...)```,
                                             scans for the bit in the reverse (decreasing index) direction.
                                             If the bit was not found, selects a new hint past the current hint.
* ```BIT_SCAN_FORWARD_MOVE_HINT_BACKWARD``` : When passed as ```scan_direction``` to
                                             ```find_any_set_near(...)``` or ```find_any_reset_near(...)```,
                                             scans for the bit in the forward (increasing index) direction.
                                             If the bit was not found, selects a new hint before the current hint.
* ```BIT_SCAN_REVERSE_MOVE_HINT_BACKWARD``` : When passed as ```scan_direction``` to
                                             ```find_any_set_near(...)``` or ```find_any_reset_near(...)```,
                                             scans for the bit in the reverse (decreasing index) direction.
                                             If the bit was not found, selects a new hint before the current hint.

### Constructors

```c++
    Bitset(unsigned arg_size = 0u)
```
Host/Device: Construct a bitset with ```arg_size``` bits.

## Data Access Functions

```c++
  unsigned size() const
```
  Host/Device: return the number of bits.

```c++
  unsigned count() const
```
  Host: return the number of bits which are set to ```1```.

```c++
  void set()
```
  Host: set all the bits to ```1```.

```c++
  void reset();
  void clear();
```
  Host/Device: set all the bits to ```0```.

```c++
  void set(unsigned i)
```
  Device: set the ```i```'th bit to ```1```.

```c++
  void reset(unsigned i)
```
  Device: set the ```i```'th bit to ```0```.

```c++
  bool test(unsigned i) const
```
  Device: return ```true``` if and only if the ```i```'th bit is set to ```1```.

```c++
  unsigned max_hint() const
```
  Host/Device: used with ```find_any_set_near(...)``` & ```find_any_reset_near(...)``` functions.
  Returns the max number of times those functions should be call
  when searching for an available bit.

```c++
  Kokkos::pair<bool, unsigned> find_any_set_near(
      unsigned hint,
      unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const
```
  Host/Device: starting at the `hint` position,
               find the first bit set to ```1```.
  Returns a ```pair<bool, unsigned>```.  When ```result.first``` is ```true``` then
  ```result.second``` is the bit position found.  When ```result.first``` is ```false``` then
  ```result.second``` is a new hint position.
  If ```scan_direction & BIT_SCAN_REVERSE```, then the scanning for the bit happens in decreasing index order;
  otherwise, it happens in increasing index order.
  If ```scan_direction & MOVE_HINT_BACKWARDS```, then the new hint position occurs at a smaller index than ```hint```;
  otherwise, it occurs at a larger index than ```hint```.

```c++
  Kokkos::pair<bool, unsigned> find_any_unset_near(
      unsigned hint,
      unsigned scan_direction = BIT_SCAN_FORWARD_MOVE_HINT_FORWARD) const;
```
  Host/Device: starting at the `hint` position,
               find the first bit set to ```0```.
  Returns a ```pair<bool, unsigned>```.   When ```result.first``` is ```true``` then
  ```result.second``` is the bit position found. When ```result.first``` is ```false``` then
  ```result.second``` is a new hint position.
  If ```scan_direction & BIT_SCAN_REVERSE```, then the scanning for the bit happens in decreasing index order;
  otherwise, it happens in increasing index order.
  If ```scan_direction & MOVE_HINT_BACKWARDS```, then the new hint position occurs at a smaller index than ```hint```;
  otherwise, it occurs at a larger index than ```hint```.

```c++
  constexpr bool is_allocated() const
```
  Host/Device: the bits are allocated on the device.

<br/>


# `ConstBitset`

## Interface

```c++
template <typename Device>
class ConstBitset
```

### Parameters

 * ```Device``` : Device that physically contains the bits.

### Constructors / assignment

```c++
  ConstBitset()
```
  Host/Device: Construct a bitset with no bits.

```c++
  ConstBitset(ConstBitset const& rhs) = default
  ConstBitset& operator=(ConstBitset const& rhs) = default
```
  Copy constructor/assignment operator.

```c++
  ConstBitset(Bitset<Device> const& rhs)
  ConstBitset& operator=(Bitset<Device> const& rhs)
```
  Host/Device: Copy/assign a ```Bitset``` to a ```ConstBitset```.

```c++
  unsigned size() const
```
  Host/Device: return the number of bits.

```c++
  unsigned count() const
```
   Host/Device: return the number of bits which are set to ```1```.

```c++
  bool test(unsigned i) const
```
  Host/Device: Return ```true``` if and only if the ```i```'th bit set to ```1```.


## NonMember Functions

```c++
template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, Bitset<SrcDevice> const& src)
```
Copy a ```Bitset``` from ```src``` on ```SrcDevice``` to ```dst``` on ```DstDevice```.

```c++
template <typename DstDevice, typename SrcDevice>
void deep_copy(Bitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src)
```
Copy a ```ConstBitset``` from ```src``` on ```SrcDevice``` to a ```Bitset``` ```dst``` on ```DstDevice```.

```c++
template <typename DstDevice, typename SrcDevice>
void deep_copy(ConstBitset<DstDevice>& dst, ConstBitset<SrcDevice> const& src)
```
Copy a ```ConstBitset``` from ```src``` on ```SrcDevice``` to ```dst``` on ```DstDevice```.

