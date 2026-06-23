PRs
===

The goal of the pull-request and review process is to ensure that the proposed change is useful and maintainable over the long run. Submitters should consider and reviewers should evaluate the below criteria.

PR Description
---------------

- Have a meaningful title: it's easier when creating the changelog or when searching through old PRs
- New features add code that needs to be maintained and tested, and risk introducing bugs. You need to motivate why the PR should be merged
- Explain what the PR does in the description: it makes it easier/faster to review
- Make the PR as small as possible: it's much easier to review five PRs 200 lines each than one single PR with 1000 lines
- If a PR is known to create conflict with other active PRs, try to coordinate, and link to each other
  - If appropriate, explain the desired review/merge order
- For complex changes that will require multiple PRs, create an issue to keep track of what has been done and what's left to do.
- Is the PR focused on a single bugfix or feature?
  - Consider moving self contained changes to separate PRs

Public Interfaces
-----------------

- Are there comprehensive tests
- Does the interface meet the use case - and for that matter is there a use case description
  - Is the usage intuitive
  - Do we really want this as a public interface, and maintain it?
- Is the interface API consistent with existing ones?
  - E.g. if everything else takes execution space instances as the first argument, don't make it the last argument
  - Does the naming style match other Kokkos APIs
- Are the interface semantics consistent with existing ones?
  - If everything else (or at least the majority) of interfaces taking an execution space instance argument is async, then new interfaces taking one should be too
- For any routine that's going to launch any parallel work, does it have an overload that takes an execution space instance, and does that overload use the instance for all parallel work?
- If the functionality is similar to an ISO C++ functionality, is the interface and behavior similar, and in places where it's not is it a conscious decision
- Is there a corresponding API documentation PR
- Are the corner cases handled (works correctly, won't compile, detected at run time)?
- Do the C++ defaulted functions do the right thing (including if needed to be marked with KOKKOS_DEFAULTED_FUNCTION)?


Internal Implementation
-----------------------

- Is the implementation style consistent with the rest of Kokkos
- Is there unnecessary code duplication
  - In particular: is code that can be shared across backends shared?
- Did it go in the right sub-part of Kokkos (core, algorithms, containers etc.)
- Are there debug checks we should add?
  - Like do the arguments make sense together etc.
- Are implementation details accidentally exposed?
- Is there unnecessary fencing
- Are there unnecessary allocations/deallocations
- Avoiding tangle of inclusions: are headers/files including only what is needed?
- Is the code expressing intent clearly (choosing expressive names for variables, functions, etc)?
- Are changes in the code and intent properly captured in the description of the PR?
- Consider appropriateness of tests for implementation details
  - We want to avoid needing to touch many tests for changes in on internal details

Tests
---------------

- For bug fix PRs: add test which would catch the issue without the fix
- Do newly added tests have the correct granularity?
- Do tests have a suitable runtime or are unnecessarily large?
