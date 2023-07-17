# Kokkos Project Planning

## Requirements Gathering

There are three requirement categories for the Kokkos Core project:

- support all relevant compute platforms, at the time of their fielding
- provide programming model features enabling performance portability
- enable an on-ramp to future ISO C++ features

A separate overarching requirement is stability of the Kokkos API.

All specific requirements are recorded and tracked in GitHub issues.

### Kokkos API Stability

Kokkos robustness and API stability is ensured through test driven development and an
explicit deprecation and removal process of existing features.

If an existing capability is determined to be outdated, or not useful anymore the Kokkos
team will deprecate the feature and thus mark it for removal at the next majpr release (occuring every two years).
A Kokkos configure option furthermore allows the defactor removal of deprecated features, enabling
customers to test whether they don't rely on features which are deprecated.

The deprecation removal cycle provides a minimum of 6 months warnings to users.
During the deprecation phase customer feedback allows for a revision of the decision.

##### Activities to support this requirement:

- provide complete testing for existing features, to ensure no accidental breakage
- evaluate features for continued usefulness and fundamental defects
- mark features as deprecated when put on the deprecation/removal path
- remove deprecated feature only at major release version change

### Platform Support

The primary requirement for the Kokkos project is to provide a robust performance-portability solution
for current and upcoming computing platforms.
The goal is to enable a seamless transition of codes between systems, and avoid situations where existing
Kokkos based code can not leverage a desired compute platform of one of our customers.

In order to meet this requirement, the Kokkos team has to anticipate new hardware platforms, before these
are fielded by customers.
The Kokkos project also needs to verify functionality with updated software stacks (compilers, runtime libraries)
on platforms as soon as they become available.

Thus the  Kokkos team must engage with hardware vendors in CoDesign efforts both independently and in conjunction
with sponsor system procurement efforts.

##### Activities to support this requirement:

- participate in facility system procurement efforts
- monitor system software stack releases from vendors (AMD, Intel, NVIDIA, HPE)
- procure new test systems where necessary
- update testing processes to account for new software stacks

### Programming Model Capabilities

Requirements for the Kokkos project are gathered from both customers, and research efforts conducted by Kokkos team members.

Customer requirements are gathered via the Kokkos Slack channel, GitHub issues, Hackathons and at usergroup meetings.
Kokkos team member assigned to a feature request will gather details of the use-case and perform an initial evaluation
of the features general applicability.
The findings will be reported and discussed at the Kokkos developer meeting, enabling a decisions on whether the feature
will be included in the roadmap.

New capability requirements by Kokkos team members are developed in separate research efforts, which prove out functinality,
use cases, and general applicability.
They are then presented to the entire Kokkos team, and discussed for inclusion in the main project.
These discussions lead to a decision on where a feature should go, wether it is important enough to be included in the primary core package,
or whether it should live as a separate library in its own repository under the Kokkos GitHub organization.


##### Activities to support this requirement:

- monitor Slack chanel and GitHub issues for new feature requests
- participate in Hackathons organized by the HPC community
- organize bi-annual Usergroup meeting
- discuss proposed features at developer meeting for inclusion into roadmap

### ISO C++ Compatibility

A third requirement for Kokkos is to provide an on-ramp for future ISO C++ standards, as well as influence where the standard goes.
This requirement serves the long term sustainability goals of Kokkos by enabling inclusion of Kokkos capabilities into ISO C++ and
thus share the maintenance burden with the entire C++ committee in the long run.

To enable the on-ramp, Kokkos will provide backports of ISO C++ features to prior C++ standards where appropriate and desired.
Kokkos will also provide versions of ISO C++ features which work on GPUs, something which is not available by default.

Kokkos features which have proven themselve, and are of interest to a wide audience are evaluated for possible inclusion in the ISO C++ standard.
The Kokkos team will write proposals for the ISO C++ committee where appropriate.

If a feature is included in the ISO C++ standard, the Kokkos team will make the API variants provided in the future C++ standard
available on currently Kokkos supported software stacks to the greates extent possible.

##### Activities to support this requirement:

- monitor requests for ISO C++ features to be provided by Kokkos
- write proposals for ISO C++ for mature Kokkos features with wide applicability
- backport relevant future ISO C++ features to standards supported by Kokkos

## Release Planning

Kokkos releases are based on the "catch the train" model - i.e. the primiary goal is to have regular releases,
not a specific feature list for each release.

Major releases happen every two years, minor releases are aimed at every 4 months, with an additional patch release in-between.

The primary difference between a major and a minor release is that deprecated features are removed at major releases, and
major release come with a bump in compiler version requirements, and potentially an updated minimum ISO C++ standard version.
Other than that there is no difference in planning and execution of major and minor releases.

The difference of patch releases to minor and major releases is that patch releases  will generally only contain bugfixes, and not new capabilities.

At the beginning of a release cycle the Kokkos Core leadership will determine high priority thrusts for the release cycle.
Furthermore, each team member will make a list of their personal priorities for the release cycle.
The priorities are discussed and refined at the Kokkos developer meeting and collected in internal documents.

Issues for each item are assigned to the Kokkos project plan https://github.com/orgs/kokkos/projects/1 including with team member assignment.

## Issue Prioritization



## Developer Coordination

## Release Process

The release process has five steps:

- create release candidate branch
- perform integration tests with release candidate
- resolve issues and cherry-pick fixes to release candidate
- check Changelog
- tag a release
- conduct release briefing for user community

When nearing a desired release date, the release candidate branch will created from the Kokkos develop branch.
Before creating the release candidate, possible delay reasons will be discussed at the developer meeting.
This could include important bug fixes, or an important feature being in the last phase of code review.

After creating the release candidate branch integration testing is started.
This includes internal testing by the Kokkos team with selected customer codes, as well as partnering
with some primary customers which will try the release candidate in their testing processes.

The release candidate creation is also announced on the Slack channel, inviting the general Kokkos
user community to test it, and provide feedback.

Defect reports (both functinality and performance) are collected as GitHub issues and marked with
"Blocks Promotion".
These items are then assigned to Kokkos team members at highest priority.

Defect resolutions are merged into the develop branch first, and then cherry picked onto the
release candidate branch, ensuring that no regression remains unaddressed on the primary development
branch.

Upon resolution of all defect reports the release candidate branch is used to create a GitHub release tag,
after checking and merging the Changelog.

After the release is created a Release Briefing date is set approximately two to three weeks after the release,
providing an overview of new capabilities to users.
The release briefing also serves as an additional point for feedback collection.

