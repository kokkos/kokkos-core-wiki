PRs and Reviews
===============

The goal of a review is to ensure that the proposed change is useful and maintainable over the long run. Submitters should consider and reviewers should evaluate the below criteria.

PR Description
---------------

- Have a meaningful title: it's easier when creating the changelog or when searching through old PRs
- Motivate why we should merge the PR: adding/changing code risks introducing a new bug. IMO one person asking for a nice to have feature doesn't qualify.
- Explain what the PR does in the description: it makes it easier/faster to review
- Make the PR as small as possible: it's much easier to review five PRs 200 lines each than one single PR with 1000 lines 
- If a PR is known to create conflict with other active PRs, try to coordinate, and link to each other
  - if appropriate, explain the desired review/merge order 
- For complex changes that will require multiple PRs, create an issue to keep track of what has been done and what's left to do.
- Is the PR focused on a single bugfix or feature?
  - consider moving self contained changes to separate PRs 

Public Interfaces
-----------------

- are there comprehensive tests
- does the interface meet the use case - and for that matter is there a use case description
  - is the usage intuitive
  - do we really want this as a public interface, and maintain it?
- is the interface API consistent with existing ones?
  - e.g. if everything else takes execution space instances as the first argument, don't make it the last argument
  - does the naming style match other Kokkos APIs
- are the interface semantics consistent with existing ones?
  - if everything else (or at least the majority) of interfaces taking an execution space instance argument is async, then new interfaces taking one should be too
- For any routine that's going to launch any parallel work, does it have an overload that takes an execution space instance, and does that overload use the instance for all parallel work?
- if the functionality is similar to an ISO C++ functionality, is the interface and behavior similar, and in places where it's not is it a conscious decision
- is there a corresponding API documentation PR
- Are the corner cases handled (works correctly, won't compile, detected at run time)?
- Do the C++ defaulted functions do the right thing (including if needed to be marked with KOKKOS_DEFAULTED_FUNCTION)?


Internal Implementation
-----------------------

- Format proposed code using clang utility, "clang-format -i," to be consistent with the Kokkos code base
- is there unnecessary code duplication
  - in particular: is code that can be shared across backends shared?
- did it go in the right sub-part of Kokkos (core, algorithms, containers etc.)
- are there debug checks we should add?
  - like do the arguments make sense together etc.
- are implementation details accidentally exposed?
- is there unnecessary fencing
- are there unnecessary allocations/deallocations
- Avoiding tangle of inclusions: are headers/files including only what is needed?
- Is the code expressing intent clearly (choosing expressive names for variables, functions, etc)?
- Are changes in the code and intent properly captured in the description of the PR?
- consider appropriateness of tests for implementation details
  - we want to avoid needing to touch many tests for changes in on internal details

Tests
---------------

- For bug fix PRs: add test which would catch the issue without the fix
- Do newly added tests have the correct granularity?
- Do tests have a suitable runtime or are unnecessarily large?

Reviewer Behavior
-----------------

- provide timely feedback and respond to changes by the author of the pull request in a reasonable amount of time; it's best to give feedback to pull requests as quickly as possible.
- only request changes if they are ready to resolve the request upon changes by the author of the pull request; stalling pull requests for requested changes that have been addressed is a problem.
- only review pull requests that have been marked as ready; we have a bunch of pull requests that explore the feasibility of ideas and just need the CI to run. Similarly, pull requests should only be marked as "ready for review" if the author is reasonably happy with the status. If the author mostly seeks feedback on general design and direction, this should be clearly communicated in the pull request description (either "draft" or "ready for review").
- mirror communication with pull request author outside of pull requests (on slack, in person, video calls, etc.) in comments to the pull request.
- contact authors directly if more clarification is needed.
- not be afraid of reviewing pull requests even if they are (slightly) outside their comfort zone.
- work with authors to bring issues/questions that need a quorum/discussion with a larger audience to the developer meeting.
