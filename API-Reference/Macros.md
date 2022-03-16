# Macros
## Execution Spaces

The following macros can be used to test whether or not a specified execution space
is enabled. They can be tested for existence (e.g. `#ifdef KOKKOS_ENABLE_SERIAL`).

| Macro                        | Description                                                                 |
| ---------------------------- | --------------------------------------------------------------------------- |
| `KOKKOS_ENABLE_SERIAL`       | Defined if the [`Serial`](Kokkos%3A%3ASerial) execution space is enabled.   |
| `KOKKOS_ENABLE_OPENMP`       | Defined if the [`OpenMP`](Kokkos%3A%3AOpenMP) execution space is enabled.   |
| `KOKKOS_ENABLE_OPENMPTARGET` | Defined if the experimental `OpenMPTarget` execution space is enabled.      |
| `KOKKOS_ENABLE_THREADS`      | Defined if the [`Threads`](Kokkos%3A%3AThreads) execution space is enabled. |
| `KOKKOS_ENABLE_CUDA`         | Defined if the [`Cuda`](Kokkos%3A%3ACuda) execution space is enabled.       |
| `KOKKOS_ENABLE_HIP`          | Defined if the experimental `HIP` execution space is enabled.               |
| `KOKKOS_ENABLE_HPX`          | Defined if the [`HPX`](Kokkos%3A%3AHPX) execution space is enabled.         |
| `KOKKOS_ENABLE_MEMKIND`      | Defined if the experimental `HBWSpace` execution space is enabled.          |
| `KOKKOS_ENABLE_SYCL`         | Defined if the experimental `SYCL` execution space is enabled.              |

## Architectures

| Macro | Description |
| ------| ---------------|
| `KOKKOS_ARCH_SSE42` | Optimize for SSE 4.2 |
| `KOKKOS_ARCH_ARMV80` | Optimize for ARMv8.0 Compatible CPU (HOST) |
| `KOKKOS_ARCH_ARMV8_THUNDERX` | Optimize for ARMv8 Cavium ThunderX CPU (HOST) |
| `KOKKOS_ARCH_ARMV81` | Optimize for ARMv8.1 Compatible CPU (HOST) |
| `KOKKOS_ARCH_ARMV8_THUNDERX2` | Optimize for ARMv8 Cavium ThunderX2 CPU (HOST) |
| `KOKKOS_ARCH_AMD_AVX2` | Optimize for AVX2 (enabled for Zen) |
| `KOKKOS_ARCH_AVX` | Optimize for AVX |
| `KOKKOS_ARCH_AVX2` | Optimize for AVX2 |
| `KOKKOS_ARCH_AVX512XEON` | Optimize for Skylake(AVX512) |
| `KOKKOS_ARCH_KNC` | Optimize for Intel Knights Corner Xeon Phi (HOST) |
| `KOKKOS_ARCH_AVX512MIC` | Optmize for Many Integrated Core (MIC; AVX512) |
| `KOKKOS_ARCH_POWER7` | Optimize for IBM POWER7 CPUs (HOST) |
| `KOKKOS_ARCH_POWER8` | Optimize for IBM POWER8 CPUs (HOST) |
| `KOKKOS_ARCH_POWER9` | Optimize for IBM POWER9 CPUs (HOST)|
| `KOKKOS_ARCH_INTEL_GEN` | Optimize for Intel GPUs Gen9+ (GPU) |
| `KOKKOS_ARCH_INTEL_DG1` | Optimize for Intel Iris XeMAX GPU (GPU) |
| `KOKKOS_ARCH_INTEL_GEN9` | Optimize for Intel GPU Gen9 (GPU)|
| `KOKKOS_ARCH_INTEL_GEN11` | Optimize for Intel GPU Gen11 (GPU) |
| `KOKKOS_ARCH_INTEL_GEN12LP` | Optimize for Intel GPU Gen12LP (GPU) |
| `KOKKOS_ARCH_INTEL_XEHP` | Optimize for Intel GPU Xe-HP (GPU) |
| `KOKKOS_ARCH_INTEL_GPU` | Set if any Intel GPU architecture has been enabled |
| `KOKKOS_ARCH_KEPLER` | Set if any NVIDIA Kepler architecture has been enabled |
| `KOKKOS_ARCH_KEPLER30` | Optimize for NVIDIA Kepler generation CC 3.0 (GPU) |
| `KOKKOS_ARCH_KEPLER32` | Optimize for NVIDIA Kepler generation CC 3.2 (GPU) |
| `KOKKOS_ARCH_KEPLER35` | Optimize for NVIDIA Kepler generation CC 3.5 (GPU) |
| `KOKKOS_ARCH_KEPLER37` | Optimize for NVIDIA Kepler generation CC 3.7 (GPU) |
| `KOKKOS_ARCH_MAXWELL` | Set if any NVIDIA Maxwell architecture has been enabled |
| `KOKKOS_ARCH_MAXWELL50` | Optimize for NVIDIA Maxwell generation CC 5.0 (GPU) |
| `KOKKOS_ARCH_MAXWELL52` | Optimize for NVIDIA Maxwell generation CC 5.2 (GPU) |
| `KOKKOS_ARCH_MAXWELL53` | Optimize for NVIDIA Maxwell generation CC 5.3 (GPU) |
| `KOKKOS_ARCH_PASCAL` | Set if any NVIDIA Pascal architecture has been enabled  |
| `KOKKOS_ARCH_PASCAL60` | Optimize for NVIDIA Pascal generation CC 6.0 (GPU) |
| `KOKKOS_ARCH_PASCAL61` | Optimize for NVIDIA Pascal generation CC 6.1 (GPU) |
| `KOKKOS_ARCH_VOLTA` | Set if any NVIDIA Volta architecture has been enabled |
| `KOKKOS_ARCH_VOLTA70` | Optimize for NVIDIA Volta generation CC 7.0 (GPU) |
| `KOKKOS_ARCH_VOLTA72` | Optimize for NVIDIA Volta generation CC 7.2 (GPU) |
| `KOKKOS_ARCH_TURING75` | Optimize for NVIDIA Turing generation CC 7.5 (GPU) |
| `KOKKOS_ARCH_AMPERE` | Set if any NVIDIA Ampere architecture has been enabled |
| `KOKKOS_ARCH_AMPERE80` | Optimize for NVIDIA Ampere generation CC 8.0 (GPU) |
| `KOKKOS_ARCH_AMPERE86 ` | Optimize for NVIDIA Ampere generation CC 8.6 (GPU) |
| `KOKKOS_ARCH_AMD_ZEN` | Optimize for AMD Zen architecture (HOST) |
| `KOKKOS_ARCH_AMD_ZEN2` | Optimize for AMD Zen2 architecture (HOST) |
| `KOKKOS_ARCH_AMD_ZEN3` | Optimize for AMD Zen3 architecture (HOST) |
| `KOKKOS_ARCH_VEGA` | Set if any AMD Vega GPU architecture as been enabled |
| `KOKKOS_ARCH_VEGA900` | Optimize for AMD GPU MI25 GFX900 (GPU) |
| `KOKKOS_ARCH_VEGA906` | Optimize for AMD GPU MI50/MI60 GFX906 (GPU) |
| `KOKKOS_ARCH_VEGA908` | Optimize for AMD GPU MI100 GFX908 (GPU) |
| `KOKKOS_ARCH_VEGA90A` | Optimize for AMD GPU MI200 GFX90A (GPU) |
