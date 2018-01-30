## Scope of Kokkos Testing

Software testing accompanies all development activities and is performed by each Kokkos developer in their work flow. Daily tasks include implementation of new features, debugging, development of test problems, user support and documentation. A critical consideration for Kokkos implementation in a host code is performance improvement; this necessitates identification of and improvement of an algorithms performance characteristics through testing.  It is considered a “failure” if an algorithm leads to a slow-down in performance. In addition, the quality/integrity of Kokkos software is preserved by creating levels of approval necessary to add-to or modify the software. An important requirement for committing to the development branch of Kokkos (develop) is code review and algorithm testing; all tests must pass at a designated level. (own hardware? a few platforms? all platforms?) The most rigorous testing is conducted for promotion of a fully-tested and clean develop branch into the Kokkos host software project, i.e, Trilinos. The Kokkos Promotion Testing section (to follow) is devoted to this most important exercise in the Kokkos workflow.

This section of the Testing description develops the primary focus of the document, the Kokkos Testing Workflow. It includes the multiple testing tasks the Kokkos team performs as highlighted above. Two important aspects of the workflow are described first as these are the foundational elements of a developer’s daily tasks; these are the Git Terminology and Workflow Branches and the Kokkos Test Scripts. Following these sections, Developer Daily Tasks can be described more easily. Finally, the “dreaded” Promotion Testing is described in detail as this task must be performed exactly as described for a successful promotion of bug fixes and new features into the hands of host code developers.

## Git Repository Terminology and Kokkos Branches

The Kokkos team uses two main Git branches in its software development process: Kokkos Master and develop. Other branches may exist but during a development cycle but the above are the primary branches of interest. Other development branches are committed to the develop branch with a pull request to become integrated into this set of working changes for the development cycle

## Description of the Kokkos Testing Scripts

These scripts are those presented (only) in the Test Scripts section above.

## Kokkos Developmental Testing (Daily Tasks)

## Kokkos Promotion Testing

Promotion testing is performed every 4 to 8 weeks.

