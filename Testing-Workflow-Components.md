The software test program components identified in the __Introduction__ are described individually in this section.

## Platforms

To achieve the goal of enabling software applications to become performant on variety of systems, software testing must take place on a wide range of architectures. Integral to this effort at Sandia is the collection of machines referred to as the Heterogeneous Advanced Architecture Platforms or HAAPs ( https://snl-wiki.sandia.gov/display/HAAPs ). A subset of these machines are identified in Table 2.1 as the primary (P) test platforms, while other machines with similar characteristics are identified as alternate (or secondary (S) ) test machines that can be used in the event of maintenance or other outages of the primary. The HAAPs link above (scroll to the Platforms section) is the official specifications for these machines. The Platforms table on the HAAPs link contains specifications for both the CPUs and the GPUs (accelerators) when present on a particular platform. Note, the information in Table 2.1 is a subset of that presented in the HAAPs table. The last column designates the network on which a particular machine exists (OHPC – Open Network, SRN – Sandia Restricted Network).

[ decide if summary needed or not: A summary of these machines in Attachment M provides further details. ]


<h4>Table 2.1: Kokkos Test Platforms and Characteristics</h4>
  
 Platform | Category | CPU Type | Nodes/Cores | Accelerator Type | Num GPUs | Network
 :--- |:--- |:--- |:--- |:--- |:--- |:---
`Apollo`| P | ? |  32  |  None |  NA  | SRN
`Bowman`| P | Knights Landing |  32/-  |  None |  NA  | OHPC 
`Ellis`| S | Knights Landing |  32/-  |  None |  NA  | SRN
`Hansen`| S | Intel  Xeon Haswell E5-2698 |  3/16  |  None |  NA  | OHPC
`Kokkos_Dev`| P | x86 |  20  |  None |  NA  | SRN
`Ride`| S | P8-Tuleta, P8-Firestone, P8-Garrison  |  4/10, 11/10, 8/10  |  NVIDIA Tesla |  4 K40, 11 K80, 8 P100  | SRN
`Shepard`| P | Intel Xeon Haswell E5-2698  |  36/16  |  None |  NA  | SRN
`Sullivan`| S | Cavium ThunderX v1, v2 |  20, 2  |  None |  NA  | OHPC
`Local Macs`| S | ? | 4 or 8  |  None |  NA  | Local
`White`| P | P8-Tuleta, P8-Firestone, P8-Garrison  |  9/10, 8/8, 8/8  |  NVIDIA Tesla |  7 K40, 7 K80, 8 P100  |  32  |  None |  NA | OHPC
`Others from Jenkins List`| P | ? |  *  |  None |  NA  | OHPC
