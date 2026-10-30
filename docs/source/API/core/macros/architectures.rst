Architectures
=============

``KOKKOS_ARCH_*`` macros record which hardware architecture Kokkos was
configured to target.

.. warning::

   Do not make your code depend on ``KOKKOS_ARCH_*`` macros.  Kokkos handles
   architecture-specific optimizations internally, and these macros tie your
   code to a particular build configuration.  They are documented here for
   completeness, and no stability guarantee is made about them.

Relationship with CMake options
-------------------------------

Configuring Kokkos with ``-DKokkos_ARCH_<NAME>=ON`` defines the macro
``KOKKOS_ARCH_<NAME>``.  For example, ``-DKokkos_ARCH_AMPERE80=ON`` defines
``KOKKOS_ARCH_AMPERE80``.

A few architectures use a different macro name.  For instance, Kokkos maps the
AMD Zen keywords ``Kokkos_ARCH_ZEN``, ``Kokkos_ARCH_ZEN2``, ...
``Kokkos_ARCH_ZEN5`` to ``KOKKOS_ARCH_AMD_ZEN``, ``KOKKOS_ARCH_AMD_ZEN2``, ...
``KOKKOS_ARCH_AMD_ZEN5`` at configuration time.

The complete list of supported architectures is in the :ref:`Architectures
section of the Configuration Guide <keywords_arch>`.  When in doubt, check the
``KokkosCore_config.h`` header generated at install time for the macros defined
for your build.

Macros without a CMake counterpart
----------------------------------

``KOKKOS_ARCH_AMD_GPU``
   Defined as a string holding the architecture flag for AMD GPUs
   (e.g. ``"gfx90A"`` when targeting an MI250X accelerator).
