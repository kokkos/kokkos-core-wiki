The software test program components identified in the __Introduction__ are described individually in this section.

## Platforms

To achieve the goal of enabling software applications to become performant on variety of systems, software testing must take place on a wide range of architectures. Integral to this effort at Sandia is the collection of machines referred to as the Heterogeneous Advanced Architecture Platforms or HAAPs ( https://snl-wiki.sandia.gov/display/HAAPs ). A subset of these machines are identified in Table 1 as the primary test or platforms; other machines with similar characteristics are identified as alternate test machines that are used in the event of maintenance or other outages of the primary. Note that the HAAPs link above (scroll to Platforms) is the official specifications for these machines. The Platforms table on the HAAPs link contains specifications for both the CPUs and the GPUs (accelerators) when present on a particular platform. Note, the information in Table 1 is a subset of that presented in the HAAPs table. The second column in Table 1 identifies whether a machine is the primary (P) or secondary (S) test platform of that type, and the last column designates the network on which a particular machine exists (OHPC – Open Network, SRN – Sandia Restricted Network). 

<h4>Table 2.1: Kokkos Test Platforms and Characteristics</h4>
  
 Platform | Category | CPU Type | Num Nodes | Accelerator Type | Num GPUs | Network
 :--- |:--- |:--- |:--- |:--- |:--- |:---
`Apollo`| P | ? |  32  |  None |  NA  | SRN
`Bowman`| P | Knights Landing |  32  |  None |  NA  | OHPC 
`Ellis`| S | Knights Landing |  32  |  None |  NA  | SRN
`Hansen`| S | Knights Landing |  32  |  None |  NA  | SRN
`Kokkos_Dev`| P | x86 |  32  |  None |  NA  | SRN
`Ride`| S | Knights Landing |  32  |  None |  NA  | SRN
`Shepard`| P | Knights Landing |  32  |  None |  NA  | SRN
`Sullivan`| S | Knights Landing |  32  |  None |  NA  | SRN
`Local Macs`| S | Knights Landing |  32  |  None |  NA  | SRN
`White`| P | Knights Landing |  32  |  None |  NA  | SRN
`Others from Jenkins List`| P | ? |  *  |  None |  NA  | OHPC

## 4.2 Using Kokkos' Makefile system