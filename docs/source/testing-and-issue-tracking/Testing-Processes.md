# Kokkos Testing Processes and Change Process

Kokkos testing falls into three categories:

 - Pull Request Testing
 - Nightly Testing
 - Integration Testing (Release Testing)
 
## Pull Request Testing

All changes to Kokkos are introduced via pull requests against the github.com develop branch of Kokkos. 
Pull requests are tested using GitHub actions workflow, as well as external testing servers.

In order to be merged two conditions must be met:

1) Automatic testing of the pull request must pass.
2) Two Kokkos core developer must approve the pull request, after checking the changes for alignment with Kokkos developer standards. 

The tested configurations in Pull Request testing cover the major deployment systems
and are executed via jenkins and travis at various institutions.

New test configurations are proposed to the Kokkos team in its developer meeting.
Inclusion of new configurations is decided based on test resource availability,
length of the entire testing pipeline, and primary computing facility software stacks.

Pull request testing also includes verification that the formatting meets 
the clang-format style specified in the repository. 

Test configurations are defined in the `kokkos/.jenkins`, and `kokkos/.github/workflows/*`  files, these files determine the official
primary software stack support.
The tested compiler versions are also listed [here](https://kokkos.github.io/kokkos-core-wiki/requirements.html).
These test configurations (sparsely) cover the cross product of hardware platforms (e.g. NVIDIA. Intel, and AMD),
compilers (e.g. GCC, Clang, NVC++), C++ standards (17-23), Kokkos backends (e.g. Cuda, OpenMP, and HIP) and Kokkos
configuration options (e.g. Debug, Relocatable Device Code).

The clang-format style file is `kokkos/.clang-format`.

Only the primary Kokkos maintainers can merge pull requests, they have the responsibility to judge whether conducted reviews meet the desired thoroughness.

## Nightly Testing

Nightly testing covers a wider range of compilers and configuration of Kokkos
on an extensive list of platforms.

All participating institutions are invited to perform nightly testing.
Test configurations are given in `kokkos/scripts/testings/` in institution specific test configuration files.

Each institution designates a test POC, who will report failures to the entire Kokkos team,
and file github issues with reproduction steps.

## Integration Testing (Release Testing)

In order for a new Kokkos version to be released integration testing is performed.
Integration testing configurations are determined and maintained by the customer projects.

This testing has three components:

#### Internal Integration Testing

Kokkos team members will perform integration testing with a select number of customer codes, they are directly involved with.
Currently that includes two code bases:

- Trilinos
- ArborX

Trilinos in particular consists of several million lines of code over multiple packages.
Both codes are tested on the primary hardware platforms, and possibly multiple software stacks (compilers in particular).
They are also tested with a limited set of configurations during nightly testing, allowing the Kokkos team to catch issues early.

#### Preferred Customer Testing

Customers funded by the same agencies as Kokkos are explicitly asked to test the release candidate before the actual release, and provide feedback.
This includes currently NNSA and Office of Science DOE users, specifically:

- SNL Empire
- SNL LAMMPS
- SNL Sparta
- SNL Sierra - Aria
- ORNL Cabana
- ANL PETSc

#### General Community testing

The release candidate is publicly available as a GitHub branch, and is advertised on the Kokkos Slack channel.
Any user of Kokkos is encouraged to test the release candidate and provide feedback.
The testing phase is at least two weeks.


