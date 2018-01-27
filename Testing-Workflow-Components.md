The software test program components identified in the __Introduction__ are described individually in this section.

## Repository Management

The Kokkos software is hosted in a repository on Github (https://github.com/github/kokkos/kokkos.git). A git clone command will by default place all software in directories with a top-level named kokkos.  Scripts, tests, description files, etc. are referenced in following texts relative to this top-level.

## Platforms

To achieve the goal of enabling software applications to become performant on variety of systems, software testing must take place on a wide range of architectures. Integral to this effort at Sandia is the collection of machines referred to as the Heterogeneous Advanced Architecture Platforms or HAAPs ( https://snl-wiki.sandia.gov/display/HAAPs ). A subset of these machines are identified in Table 2.1 as the primary (P) test platforms, while other machines with similar characteristics are identified as alternate (or secondary (S) ) test machines that can be used in the event of maintenance or other outages of the primary. The HAAPs link above (scroll to the Platforms section) is the official specifications for these machines. The Platforms table on the HAAPs link contains specifications for both the CPUs and the GPUs (accelerators) when present on a particular platform. Note, the information in Table 2.1 is a subset of that presented in the HAAPs table. The last column designates the network on which a particular machine exists (OHPC – Open Network, SRN – Sandia Restricted Network).

[ decide if summary needed or not: A summary of these machines in Attachment M provides further details. ]


<h4>Table 2.1: Kokkos Test Platforms and Characteristics</h4>
  
 Platform | Category | CPU Type | Nodes/Cores | Accelerator Type | Num GPUs | Network
 :--- |:--- |:--- |:--- |:--- |:--- |:---
`Apollo`| P | ? |  ?  |  None |  NA  | Local/OHPC?
`Bowman`| P | Knights Landing |  32/-  |  None |  NA  | OHPC 
`Ellis`| S | Knights Landing |  32/-  |  None |  NA  | SRN
`Hansen`| S | Intel  Xeon Haswell E5-2698 |  3/16  |  None |  NA  | OHPC
`Kokkos_Dev`| P | x86 |  20  |  None |  NA  | SRN
`Ride`| S | P8-Tuleta, P8-Firestone, P8-Garrison  |  4/10, 11/10, 8/10  |  NVIDIA Tesla |  4 K40, 11 K80, 8 P100  | SRN
`Shepard`| P | Intel Xeon Haswell E5-2698  |  36/16  |  None |  NA  | SRN
`Sullivan`| S | Cavium ThunderX v1, v2 |  20, 2  |  None |  NA  | OHPC
`Multiple Macs`| S | ? | 4 or 8  |  None |  NA  | Local
`White`| P | P8-Tuleta, P8-Firestone, P8-Garrison  |  9/10, 8/8, 8/8  |  NVIDIA Tesla |  7 K40, 7 K80, 8 P100  |  32  |  None |  NA | OHPC
`Others (Jenkins Targets)`| P | ? |  *  |  None |  NA  | OHPC


## Batch Queues

Examining the list of machines in Table 2.1, one can see that a wide range of hardware types constitute our collection of test beds. Some platforms have several different types of processors, including CPUs and GPUs. In order to access a homogeneous collection of nodes when runs are made on one these machines, several queues have been setup, one for each hardware/architecture type. It is necessary to load the proper environment for these nodes and use a batch submission script that selects the desired queue explicitly. More details are provided in the section that discusses the test scripts used for all Kokkos testing.

## Computer Accounts

Sandia computing machines are connected to various networks and require access permissions be obtained through an account control system, normally WebCARS. Each of the machines listed above requires an account be obtained through WebCARS, except for Apollos and Kokkos-dev. Machines identified as Local are normally personal hardware of various kinds that are issued to (Kokkos team) staff members. Accounts on _Kokkos-dev_ may be obtained through permission of Kokkos project leads and the assistance of CSRI CSU staff members. The _Kokkos-dev_ and primary machines are required accounts for most testing, but most especially for **promotion testing** (described below). It is recommended that Kokkos team members obtain accounts on all the machines listed in Table 2.1.

## Compilers
 
Numerous compilers are available on the platforms identified in Table 2.1; this availability is provided by a range of modules. The modules that are utilized for the Kokkos testing described herein are identified in the Section on Test Scripts. From among the wide range of available compilers, Kokkos supports a select set of these. The final discriminator of what constitutes support is determined by the set of compilers tested regularly through the nightly jobs submitted using the Jenkins servers. A version of this list is also provided through the README.md in the Kokkos Git repository ( https://github.com/kokkos/kokkos.git  ), and, the current version is presented in Attachment C. Note, this set of compilers is not supported on any one platform but among the collection of platforms described in the Platform section above. This list also appears in the _test_all_sandia_ driver script in **kokkos/config**.

## Backends

A backend in computing terminology refers to the system component that performs the majority of the work or computation.  For application to Kokkos, a backend essentially identifies a mode in which executables are compiled and defines the manner in which the hardware/processors are addressed to complete the assigned commands. The supported options for backends in Kokkos-based simulations are:


```c++
* Serial
* OpenMP
* Threads (PThreads)
* QThreads 
* Cuda
* ROCm
```

## Test Scripts

All Kokkos testing is performed using one or more shell-scripts that are contained in the Kokkos source code; these may be found in directories kokkos/config or kokkos/scripts. These were developed to setup and run the Kokkos tests (Section Test Descriptions) on several backends for several supported compilers on several platforms and then to analyze and report the results to the test performer. These shell scripts are listed below as well as the Kokkos directory in which it is located; see Repository Management which identifies the host Github site. These scripts are not discussed or reviewed here but will be identified in the Section on Testing and the circumstances in which each is used.
```c++
* kokkos/config:
* * test-all-sandia

* kokkos/scripts
* * snapshot.py

* kokkos/scripts/testing_scripts
* * jenkins_test_driver
* * test_kokkos_master_develop_promotion.sh

* kokkos/scripts/trilinos-integration
* * checkin-test               
* * prepare_trilinos_repos.sh
* * shepard_jenkins_run_script_pthread_intel  
* * shepard_jenkins_run_script_serial_intel  
* * white_run_jenkins_script_cuda
* * white_run_jenkins_script_omp
```

## Test Descriptions

The Kokkos source directories contain the test problems that are exercised in nightly, release and promotion testing. The following kokkos source directories contain nearly a hundred tests. Please consult individual directories and test problems for necessary details. Results from these tests are presented as pass/fail.

```c++
kokkos/algorithms/unit_tests   - 8
kokkos/benchmarks    - 4
kokkos/containers/{performance_tests   -  8; unit_tests   -  14 }
kokkos/core/{perf_test   -  ?; unit_test    -  ? } 
example  -  22
```


