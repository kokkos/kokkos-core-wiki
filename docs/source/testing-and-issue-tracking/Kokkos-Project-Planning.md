# Kokkos Project Planning

## Requirements Gathering

There are four requirement categories for the Kokkos Core project:

- provide a stable, well-tested API avoiding breakage
- support all relevant compute platforms, at the time of their fielding
- provide programming model features enabling performance portability
- enable an on-ramp to future ISO C++ features

A separate overarching requirement is the stability of the Kokkos API.

All related specific actionable tasks are recorded and tracked in GitHub issues and pull requests.

### Kokkos API Stability

Robustness and API stability are ensured through test-driven development and an
explicit deprecation and removal process of existing features.

If an existing capability is determined to be outdated, or not useful anymore the Kokkos
team will deprecate the feature and thus mark it for removal in the next major release (occurring every three years).
A Kokkos configure option furthermore allows the de-facto removal of deprecated features, enabling
customers to test whether they rely on them.

The deprecation-removal cycle provides warnings for a minimum of 6 months to users.
During the deprecation phase, customer feedback allows for a revision of the deprecation decision.

#### Activities to support this requirement:

- provide complete testing for existing features, to ensure no accidental breakage
- evaluate features for continued usefulness and fundamental defects
- tag features as deprecated, when put on the deprecation/removal path
- remove deprecated features only at major release version change

### Platform Support

The primary requirement for the Kokkos project is to provide a robust performance-portability solution
for current and upcoming computing platforms.
The goal is to enable a seamless transition of codes between systems and avoid situations where existing
Kokkos-based codes cannot leverage a desired computing platform.

In order to meet this requirement, the Kokkos team has to anticipate new hardware platforms, before these
are field-tested by customers.
The Kokkos project also needs to verify functionality with updated software stacks (compilers, runtime libraries)
on platforms as soon as they become available to the Kokkos team (ideally before deployment on customer platforms).

Thus, the  Kokkos team must engage with hardware vendors in co-design efforts both independently and in conjunction
with system procurement efforts of funding agencies.

#### Activities to support this requirement:

- participate in facility system procurement efforts
- monitor system software stack releases from vendors (AMD, Intel, NVIDIA, HPE)
- engage vendors to enable testing of Kokkos with pre-release software development kits
- procure new test systems where necessary
- update testing processes to account for new software stacks

### Programming Model Capabilities

Requirements for the Kokkos project are gathered from both customers, and research efforts conducted by Kokkos team members.

Customer requirements are gathered via the Kokkos Slack channel, GitHub issues, Hackathons, and at user group meetings.
Kokkos team members assigned to a feature request will gather details of the use case and perform an initial evaluation
of the feature's general applicability.
The findings will be reported and discussed at the Kokkos developer meeting, enabling a decision on whether the feature
will be included in the roadmap.
Feature discussions will be recorded and tracked in public GitHub issues.

New capability requirements by Kokkos team members are developed in separate research efforts, which explore functionality,
use cases, and general applicability.
They are then presented to the entire Kokkos team and discussed for inclusion in the main project.
These discussions lead to a decision on where a feature should go, whether it is important enough to be included in the primary core package,
or whether it should live as a separate library in its own repository under the Kokkos GitHub organization.

#### Activities to support this requirement:

- monitor Slack channel and GitHub issues for new feature requests
- participate in Hackathons organized by the HPC community
- organize bi-annual Usergroup meeting
- discuss proposed features at developer meeting for inclusion into roadmap

### ISO C++ Compatibility

A third requirement for Kokkos is to provide an on-ramp for future ISO C++ standards, as well as influence where the standard goes.
This requirement serves the long-term sustainability goals of Kokkos by enabling the inclusion of Kokkos capabilities into ISO C++ and
thus share the maintenance burden with the entire C++ implementer community in the long run.

To enable the on-ramp, Kokkos will provide backports of ISO C++ features to prior C++ standards, where appropriate and desired.
Kokkos will also provide extensions of ISO C++ features that work on GPUs, something which is not available by default.

Kokkos features which have proven themselves, and are of interest to a wide audience are evaluated for possible inclusion in the ISO C++ standard.
The Kokkos team will write proposals for the ISO C++ committee when appropriate.

If a feature is included in the ISO C++ standard, the Kokkos team will make the API variants provided in the future C++ standard
available on currently Kokkos-supported software stacks to the greatest extent possible.

#### Activities to support this requirement:

- participate in ISO C++ committee meetings
- monitor requests for ISO C++ features to be provided by Kokkos
- write proposals for ISO C++ for mature Kokkos features with wide applicability
- backport relevant future ISO C++ features to standards supported by Kokkos

## Release Planning

Kokkos releases are based on the "catch the train" model - i.e. the primary goal is to have regular releases,
not a specific feature list for each release.

Major releases happen every three years, minor releases are aimed at every 3-4 months, with additional patch releases as necessary.

The primary difference between a major and a minor release is that deprecated features are only removed at major releases, and
major releases come with a bump in minimal compiler version requirements and an updated minimum ISO C++ standard version.
Other than that, there is no difference in the planning and execution of major and minor releases.

In contrast to major and minor releases, patch releases generally only contain bug fixes and no new capabilities.

At the beginning of a release cycle, the Kokkos Core leadership will determine high-priority thrusts for the release cycle.
Furthermore, each team member will make a list of their personal priorities for the release cycle.
The priorities are discussed and refined at the Kokkos developer meeting and collected in internal documents.

Issues for each item are assigned to the [Kokkos project plan](https://github.com/orgs/kokkos/projects/1) including team member assignments.

The [Kokkos project plan](https://github.com/orgs/kokkos/projects/1) assigns issues to one of 7 categories:

- *Unassigned:* issues that aren't assigned yet to team members.
- *Unassigned - Priority:* issues that aren't assigned yet to team members, but are high priority. These should be assigned at the next weekly developer meeting.
- *To Do:* Issue was assigned to a team member but is not yet actively worked on.
- *To Do - Priority:* Issue was assigned to a team member, but is not yet actively worked on. It is expected to be the next item in the queue of the assigned developer. If this item does not transition to *In Progress* by the next developer meeting, reassignment is considered.
- *In Progress:* Issue is getting worked on.
- *In Progress - Priority:* Issue is getting worked on. Code reviews for this issue are considered a priority, in order to get this resolved as soon as possible.
- *Done:* Issue is addressed via merged pull request, or was closed because of new information which made it obsolete. For merged pull requests it is ensured that a changelog entry was generated, if appropriate, before removing the item from the project plan.


## Issue Prioritization

Issue prioritization is performed via two avenues:
- Kokkos Leadership meeting
- General Kokkos developer meeting.

The Leadership meeting happens every week on Mondays.
It serves multiple purposes:
- determine urgent action items for the week
- go through new issue list, and triage criticality
- work through Kokkos planning items
- perform preliminary team assignments for new action items
- generate a draft for the developer meeting agenda

Prioritization of items is recorded in the [Kokkos project plan](https://github.com/orgs/kokkos/projects/1)

Meeting notes are kept in a private repository: [internal repository](https://github.com/kokkos/internal-documents)

Further issue prioritization happens at the developer meeting discussed below.

## Developer Coordination

The team primarily use the #nucleus channel on Slack to communicate.
Members are added by Christian or Damien once they have joined [Slack](https://kokkosteam.slack.com).
Developers can have both public and private conversations with each other.
They can ask questions about parts of the code they are less familiar with or
ask for feedback on any ongoing issue.
Conversations on Slack are to be considered as ephemeral.  Messages older than 90 days are deleted (unpaid plan).
If something needs to be referenceable longer term, then it needs to be discussed on GitHub wherever appropriate.
Private information may be hosted on the [internal repository](https://github.com/kokkos/internal-documents) but do not post NDA data on there.

Kokkos developer meeting held once a week on Wednesdays 2pm ET / 12 pm MT / 18:00 UTC on Zoom.
The agenda is posted on the internal repository ahead of time (it can be found under the [`meeting-notes/`](https://github.com/kokkos/internal-documents/tree/master/meeting-notes/2023) directory).
Developers are allowed to edit the agenda and add topics or issues that they would like to be discussed at the meeting.

## Release Process

The release process has six steps:

- create release candidate branch
- perform integration tests with release candidate
- resolve issues and cherry-pick fixes to release candidate
- check Changelog
- tag a release
- conduct release briefing for user community

When nearing a desired release date, the release candidate branch will be created from the Kokkos develop branch.
Before creating the release candidate, possible delay reasons will be discussed at the developer meeting.
This could include important bug fixes, or an important feature being in the last phase of code review,
but is generally done under exceptional circumstances.
Furthermore, merging major new features into the development branch may be delayed until after the creation
of the release candidate.
This ensures that major new features have a period of testing in the develop branch before they are shipped.

After creating the release candidate branch integration testing is started.
This includes internal testing by the Kokkos team with selected customer codes, as well as partnering
with some primary customers who will try the release candidate in their testing processes.

The release candidate creation is also announced on the Slack channel, inviting the general Kokkos
user community to test it, and provide feedback.

Defect reports (both functionality and performance) are collected as GitHub issues and marked with
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

