## Scope of Kokkos Testing

Software testing accompanies all development activities and is performed by each Kokkos developer in their work flow. Daily tasks include implementation of new features, debugging, development of test problems, user support and documentation. A critical consideration for Kokkos integration into a host code is performance improvement; this necessitates identification of and improvement of Kokkos algorithms performance characteristics through testing.  An algorithm is presumed to have “failed” if it leads to a slow-down in performance. An additional consideration in the daily Kokkos workflow is maintenance of the quality/integrity of Kokkos software. This is addressed by creating levels of approval that govern when and how algorithms may be modified and created. An important requirement for committing to the development branch of Kokkos (develop) is code review and algorithm testing; all tests must pass at a designated level. (own hardware? a few platforms? all platforms?) The most rigorous testing is conducted for promotion of a fully-tested and clean develop branch into the Kokkos host software project, i.e, Trilinos. The **Kokkos Promotion Testing** section (to follow) is devoted to this most important exercise in the Kokkos workflow.

This section of the Testing description develops the primary focus of the document, the Kokkos Testing Workflow. It includes the multiple testing tasks the Kokkos team performs as outlined above. Two important aspects of the workflow are described first as these are the foundational elements of a developer’s daily tasks; these are the **Git Terminology and Kokkos Branches** and the **Kokkos Test Scripts**. Following these sections, Developer Daily Tasks can be described more easily. Finally, the all-important **Promotion Testing** is presented in detail as this task must be performed exactly as described for a successful promotion of bug fixes and new features into the hands of host code developers.

## Git Repository Terminology and Kokkos Branches

In Git, a branch is a divergence from the main (or master) line of a software repository; it is where most of the development takes place. The Kokkos team uses two main Git branches in its software development process: Kokkos *Master* and *develop*. Other branches will exist during a development cycle, e.g. *issue-865* or *array-bounds*, but the above are the branches of interest when merging changes into the master repository. The other development branches (e.g., *issue-865*) are committed to the develop branch with a pull request to become integrated into the *develop* branch set of working changes for the development cycle.

## Description of the Kokkos Testing Scripts

This section provides a functional description of the scripts identified in the **Test Scripts** section above. The role of each script in the Kokkos workflow is described.


## Kokkos Developmental Testing (Daily Tasks)

## Kokkos Promotion Testing

Promotion testing is performed every 4 to 8 weeks.

