# Kokkos Testing Workflow Components

The software test program components identified in the __Introduction__ are described individually in this section. 
This chapter presents multiple aspects of the Kokkos test program and identifies the role of each component of the workflow. Establishing a “loaded vocabulary”, the importance of these components to the overall testing activities is described in this chapter. The actual testing activities are presented in the following chapter.

## Software Management for the Kokkos Git Repository

Git is a distributed management system and is used for the Kokkos software repository; a complete snapshot of the Kokkos software can be obtained from the Kokkos clone [site]( https://github.com/kokkos/kokkos.git). A clone from this Git remote site, called by default __origin__, downloads a branch called by default __master__. Each _branch_ is an independent copy of a Git database (called the software repository here) and where development of a software project takes place. The Kokkos team has two branches of primary importance in its software development process: Kokkos __master__ and __develop__; the former is treated as the official version of Kokkos (forked from Trilinos) and the latter the official branch that collects changes from developers during a development cycle. Both branches restrict write-privileges to the project owners. Developers create branches, e.g. issue-865 or array-bounds, for their work and submit pull requests (fetch and merge) to the project owners to integrate the branch changes for the current development cycle. The pull request often contains suggestions for review and presentation of testing results. When a development cycle is complete, the process for integration of Kokkos changes into the Trilinos repository (the promotion process) is begun. At the conclusion of a successful promotion, the _master_ branch of Kokkos receives a pull request from the _develop_ branch; after approval, the cycle begins again.

## Platforms

To achieve the goal of enabling software applications to become performant on a variety of systems, software testing must take place on a wide range of architectures. Integral to this effort at Sandia is the collection of machines referred to as the Heterogeneous Advanced Architecture Platforms or [HASPs](https://snl-wiki.sandia.gov/display/HAAPs). A subset of these machines are identified in Table 2.1 as the primary (P) test platforms, while other machines with similar characteristics are identified as alternate (or secondary (S) ) test machines that can be used in the event of maintenance or other outages of the primary. The HASPs link above (scroll to the Platforms section) is the official specifications for these machines. The Platforms table on the HASPs link contains specifications for both the CPUs and the GPUs (accelerators) when present on a particular platform. Note, the information in Table 2.1 is a subset of that presented in the HASPs table. The last column designates the network on which a particular machine exists (OHPC – Open Network, SRN – Sandia Restricted Network).

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

Examining the list of machines in Table 2.1, one can see a wide range of hardware types constitute our collection of test beds. Some platforms have several different types of processors, including CPUs and GPUs. In order to access a homogeneous collection of nodes when test problems are exercised on one of these machines, several queues have been setup, one for each hardware/architecture type. It is necessary to load a specific environment for these nodes and use a batch submission script that targets the specific queue explicitly. The command used to start test scripts on these platforms must contain a queue specification option.

## Computer Accounts

Sandia computing machines are connected to multiple networks and require access permissions be obtained through an account control system; account control is normally through the WebCARS intranet utility. Each of the machines listed in Table 2.1 requires an account be obtained through WebCARS, except for _Apollos_ and _Kokkos-dev_. Machines identified as _Local_ are normally personal hardware of various kinds that are issued to (Kokkos team) staff members. Accounts on _Kokkos-dev_ may be obtained through permission of Kokkos team leaders and the assistance of CSRI CSU staff members. The _Kokkos-dev_ and primary machines are required accounts for most testing, but most especially for __promotion testing__ (described in Section 3). It is recommended that Kokkos team members obtain accounts on all the machines identified in Table 2.1.

## Compilers
 
Numerous compilers are installed on the platforms identified in Table 2.1; access to these compilers is administered through a module utility and the modules setup for the installed compilers. From among this set of installed compilers, Kokkos selects a subset that will constitute __Kokkos-supported__ compilers. Every supported compiler requires extensive testing to inherit the __supported__ label. In particular, compiler-platform pairs must be include in the daily/nightly testing regimen that is established using the Jenkins continuous-integration software. All supported compilers are included in this extensive suite of testing evaluations. A list of these test jobs is only present as a view of the Jenkins dashboard. However, a version of this list can be seen as a snapshot (in-time) that exists as the _README_ in the [Kokkos Git repository](https://github.com/kokkos/kokkos.git). The current version of this list is presented in the [Attachments](#A. Identification of Compilers Supported by Kokkos). Important to remember is that this set of compilers is not supported on any one platform but as a collection of installed and tested compilers among the numerous platforms described in the __Platforms__ section above. This list is also built into the _test_all_sandia_ driver script (repository directory __kokkos/config__) that controls many of the individual developer-requested testings that happen in the Kokkos-team work flow.

## Backends

A backend in computing terminology refers to the system component that performs the majority of the work or computation.  For application to Kokkos, a backend essentially identifies a joint hardware-software configuration that enables the software application to address and utilize the computational power of specific processor types and their local memory spaces. Essential aspects of this Kokkos feature are the execution and memory spaces that define backends. The supported options for backends in Kokkos-based simulations are (correct?, please edit):

```c++
* Serial
* OpenMP
* Threads (PThreads)
* QThreads 
* Cuda
* HIP
```
Note that the HIP backend is currently experimental (under development).

## Test Scripts

All Kokkos testing is directed by shell-scripts that are a part of the Kokkos source code; these may be found in the top-level kokkos directory as well as kokkos/config and kokkos/scripts. They were developed to setup and run the Kokkos tests on several backends for several supported compilers on several platforms; the scripts then analyze and report the results to the test performer. These scripts and their role in the testing workflow are briefly described in Table 2.2 (Kokkos Test Script Descriptions).

<h4>Table 2.2: Kokkos Test Script Descriptions </h4>
  
 Script Name | Kokkos Source Directory | Script Description 
 :----: |:----: |:--- 
`generate_makefile.bash`| __kokkos__ | for selected options, this shell script generates a Makefile that builds and runs the tests identified in section __Test Descriptions__. Options to be specified include the devices (i.e., the execution space), the architecture, the compiler and compiler flags for the Kokkos library, and a few other settings.  
`test-all-sandia`| __kokkos/config__ | a shell script that is run on a select machine for a select group of compilers and for the machine architecture. The script accepts a set of options that define the specific results desired for a particular run. For the machine this script is running on, the set of available compilers for the machines architecture will be run and the results processed. It can also run “spot checks” for a select subset of compilers on machines Apollos and Kokkos-dev.
`jenkins_test_driver`| __kokkos/scripts/testing_scripts__ | This script accepts a build-type and host-compiler input to launch a build and run job on a target machine having created a makefile using _generate_makefile.bash_; it starts a job on the Jenkins continuous integration server; results are presented for configure, build and test steps; output for each of these steps can be examined through a dashboard.
`snapshot.py`| __kokkos/scripts__ | In Git terminology, a snapshot is a copy of an entire software repository at a specific point in time (defined by its SHA-1). This python script takes a snapshot of one repository and performs the necessary repository actions to merge it into another repository at a specific point in history while generating a commit message for the git process to accomplish this joining operation.
`prepare_trilinos_repos.sh` | kokkos/scripts/trilinos-integration | This script defines a set of environment variables for the two checkouts of trilinos necessary to perform the trilinos testing that precedes promotion of an updated Kokkos into the Trilinos repository. The environment variables are paths that point to the “pristine” trilinos branch and to the “updated” trilinos branch (contains a snapshot of the Kokkos develop branch into the Trilinos develop branch). This is the initial step preceding the trilinos integration tests that will take place on platforms Shepard and White.
`shepard_jenkins_run_script_pthread_intel` | kokkos/scripts/trilinos-integration | loading a particular module for platform Shepard and the set of environment variables that point to the pristine and updated trilinos branches, this script sets up and runs a pthreads version of the trilinos checkin tests. Build flags for the appropriate Kokkos backends are set and Kokkos cloned if necessary. Tests are run using scripts developed by the Sandia SEMS team; results are compared.
`shepard_jenkins_run_script_serial_intel` | kokkos/scripts/trilinos-integration | same as for script _shepard_jenkins_run_script_pthread_intel_ except it runs the _serial_ workspace in place of the _pthread_ workspace.
`white_run_jenkins_script_cuda` | kokkos/scripts/trilinos-integration | as for the trilinos tests on Shepard, these are on platform White to test different compilers and architecture. Similar setup of flags, backends and libraries precede submission of batch jobs to run using the SEMS-developed scripts. Test results are reported as for the tests on Shepard.
`white_run_jenkins_script_omp` | kokkos/scripts/trilinos-integration | same as for script _white_run_jenkins_script_cuda_ except it runs the _cuda_ workspace in place of the _omp_ workspace.
`test_kokkos_master_develop_promotion.sh` | kokkos/scripts/testing_scripts | [not used ??] For a specific set of parameters – backend, module, compiler, CXXFlags, architecture, kokkos options, cuda options, and HWLOC – a makefile is crated using generate_makefile.sh. [this makefile script is not the same as mentioned above with a .bash extension]
`checkin-test` | kokkos/scripts/trilinos-integration | this script loads a set of SEMS modules for a trilinos checkinTest. This latter test (script) does not exist in the Kokkos repository at this time.

## Kokkos Tests: Unit and Performance

As described in the __Introduction__, Kokkos is a library of macros designed to enable applications of all flavors to experience the power and speed of Next Generation computer processors in the solution of their central equations. Kokkos’ role is as an _enabler_ in these applications when it’s macros are properly integrated into the central algorithms (viz, kernels) of these applications. Testing of Kokkos’ macros is accomplished by replicating some of the mathematical implementations of typical kernels at a smaller scale and verifying the accuracy and performance characteristics of these replicas in a series of unit and performance tests.

The Kokkos source directories identified below contain nearly a hundred tests problems that are exercised in debugging, nightly, release and promotion testing. Individual directories and test problems should be examined for necessary details of each test.

```c++
kokkos/algorithms/{unit_tests   - 8; performance_tests  - ? }
kokkos/benchmarks    - 4
kokkos/containers/{performance_tests   -  8; unit_tests   -  14 }
kokkos/core/{perf_test   -  ?; unit_test    -  ? } 
example  -  22
```
The results of running these tests are reported as either __pass__ or __fail__. Test problems that fail are identified for scrutiny so that software errors, inadequacies, and/or test problem deficiencies may be located and corrected.
