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

All Kokkos testing is performed using one or more shell-scripts that are contained in the Kokkos source code; these may be found in directories kokkos/config or kokkos/scripts. These were developed to setup and run the Kokkos tests (Section Test Descriptions) on several backends for several supported compilers on several platforms and then to analyze and report the results to the test performer. These shell scripts are listed below as well as the Kokkos directory in which it is located; see Repository Management which identifies the host Github site. These scripts and their role in the testing workflow are briefly described in Table 2.2.

<h4>Table 2.2: Kokkos Test Script Descriptions </h4>
  
 Script Name | Kokkos Source Directory | Script Description 
 :----: |:----: |:--- 
`generate_makefile.bash`| __kokkos__ | for selected options, this shell script generates a Makefile to run a set of tests. Options to be specified include the devices (i.e., the execution space), the architecture, the compiler and compiler flags for the Kokkos library, and a few other settings. A Makefile is created that builds and runs the tests identified in section __Test Descriptions__.
`test-all-sandia`| __kokkos/config__ | a shell script that is run on a select machine for a select group of compilers and for the machine architecture.  The script accepts a set of options that define the specific results desired for a particular run.  For the machine this script is running on, the set of available compilers for the machines architecture will be run and the results processed. It can also run “spot checks” for a select subset of compilers on machines Apollos and Kokkos-dev.
`jenkins_test_driver`| __kokkos/scripts/testing_scripts__ | This script accepts a build-type and host-compiler input to launch a build and run job on a target machine having created a makefile using _generate_makefile.bash_; it starts a job on the Jenkins continuous integration server; results are presented for configure, build and test steps; output for each of these steps can be examined through a dashboard.
`snapshot.py`| __kokkos/scripts__ | In Git terminology, a snapshot is a copy of an entire software repository at a specific point in time (defined by its SHA-1). This python script takes a snapshot of one repository and performs the necessary repository actions to merge it into another repository at a specific point in history while generating a commit message for the git process to accomplish this joining operation.
`prepare_trilinos_repos.sh` | kokkos/scripts/trilinos-integration | This script defines a set of environment variables for the two checkouts of trilinos necessary to perform the trilinos testing that precedes promotion of an updated Kokkos into the Trilinos repository. The environment variables are paths that point to the “pristine” trilinos branch and to the “updated” trilinos branch that contains a snapshot of the Kokkos develop branch into the Trilinos develop branch. This is the initial step preceding the trilinos integration tests that will take place on platforms Shepard and White.
`shepard_jenkins_run_script_pthread_intel` | kokkos/scripts/trilinos-integration | loading a particular module for platform Shepard and the set of environment variables that point to the pristine and updated trilinos branches, this script sets up and runs a pthreads version of the trilinos checkin tests. Build flags for the appropriate Kokkos backends are set along with build flags and Kokkos cloned if necessary. Tests are run using scripts developed by the Sandia SEMS team; results are compared.
`shepard_jenkins_run_script_serial_intel` | kokkos/scripts/trilinos-integration | same as for script _shepard_jenkins_run_script_pthread_intel_ except it runs the _serial_ workspace in place of the _pthread_ workspace.
`white_run_jenkins_script_cuda` | kokkos/scripts/trilinos-integration | as for the trilinos tests on Shepard, these are on platform White to test different compilers and architecture. Similar setup of flags, backends and libraries precede submission of batch jobs to run using the SEMS-developed scripts. test results are reported as for the tests on Shepard.
`white_run_jenkins_script_omp` | kokkos/scripts/trilinos-integration | same as for script _white_run_jenkins_script_cuda_ except it runs the _cuda_ workspace in place of the _omp_ workspace.
`test_kokkos_master_develop_promotion.sh` | kokkos/scripts/testing_scripts | [not used ??] For a specific set of parameters – backend, module, compiler, CxxFlags, architecture, kokkos options, cudea options, and HWLOC – a makefile is crated using generate_makefile.sh. [this makefile script is not the same as mentioned above with a .bash extension]
`checkin-test` | kokkos/scripts/trilinos-integration | this script loads a set of SEMS modules for a trilinos checkinTest. This latter test (script) does not exist in the Kokkos repository at this time.


## Kokkos Tests: Unit and Performance

As described in the Introduction, Kokkos is a library of macros designed to enable applications of all flavors to experience the power and speed of evolving computer processors in the solution of their central equations. Kokkos’ role is as an enabler in these applications when it’s macros are properly integrated into the central algorithms (viz, kernels) of these applications. Testing of Kokkos’ macros is accomplished by replicating the mathematical implementations of typical kernels at a smaller scale and verifying the accuracy and performance characteristics of these replicas in a series of unit and performance tests.

The Kokkos source directories identified below contain nearly a hundred tests problems that are exercised in nightly, release and promotion testing. Individual directories and test problems should be examined for necessary details of each test.

```c++
kokkos/algorithms/{unit_tests   - 8; performance_tests  - ? }
kokkos/benchmarks    - 4
kokkos/containers/{performance_tests   -  8; unit_tests   -  14 }
kokkos/core/{perf_test   -  ?; unit_test    -  ? } 
example  -  22
```
The results of running these tests are reported as either __pass__ or __fail__. Test problems that fail are identified for scrutiny so that software errors, inadequacies, and/or test problem deficiencies may be located and corrected.

