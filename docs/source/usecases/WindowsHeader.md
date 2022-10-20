# Kokkos and `Windows.h`

When using Kokkos on windows a program or library might include `windows.h`. This is problematic as this header defines two macros with the names `min` and `max` unless `NOMINMAX` is defined previously.
The Preprocessor replaces strings in the source code with the macros yielding an uninterpretable result and thus compilation fails.
Therefore, the header `Kokkos_Core.hpp` is protected against these macros, meaning they are undefined at the beginning and redefined at the end of the header file.
Even though definitons inside `Kokkos_Core.hpp` are protected against the macros, code outside is not.
Thus, it is on the user to deal with the macros being defined, either by defining `NOMINMAX` before including `windows.h` or by putting `()` around names that contain `min` or `max`.
