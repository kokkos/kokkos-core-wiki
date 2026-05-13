Reviews
=======

The goal of a review is to help the code contributor to improve the code while also checking if it is the best approach to the described problem.

Reviewer Behavior
-----------------

- Provide timely feedback and respond to changes by the author of the pull request in a reasonable amount of time; it's best to give feedback to pull requests as quickly as possible.
- Only request changes if they are ready to resolve the request upon changes by the author of the pull request; stalling pull requests for requested changes that have been addressed is a problem.
- Only review pull requests that have been marked as ready; we have a bunch of pull requests that explore the feasibility of ideas and just need the CI to run. Similarly, pull requests should only be marked as "ready for review" if the author is reasonably happy with the status. If the author mostly seeks feedback on general design and direction, this should be clearly communicated in the pull request description (either "draft" or "ready for review").
- Mirror communication with pull request author outside of pull requests (on slack, in person, video calls, etc.) in comments to the pull request.
- Contact authors directly if more clarification is needed.
- Don't be afraid of reviewing pull requests even if they are (slightly) outside your comfort zone.
- Work with authors to bring issues/questions that need a quorum/discussion with a larger audience to the developer meeting.

Checklist
=========

The following checklist can serve as guidance for a thorough review. It is extensive and tries to be general, thus it might be overkill for small PRs (e.g. simple bugfixes)

Fundamental Questions
---------------------

- Is the PR title clear enough about the scope of the changes?
- Is clear what problem the PR is trying to resolve?
- Is the proposed solution appropriate for the problem?
- Is it working (check the CI)?
- Does the PR introduce/change abstractions? If yes, what behavior has changed?
- Which other abstractions/classes/concepts in Kokkos interact with the change?
- Is there a good reason for including this code?

Design
------

- Does it adhere to design principles like SOLID,DRY?
- Does the design and variable naming fit into the rest of Kokkos?
- Is the current design restricting future design choices? Does/should it allow extension?
- Is any implicit dpenendency introduced?
- Is the design reasonably simple?
- Are the used names descriptive of what something does?

Complexity
----------

- How complex is the change? Should/can it be split into multiple PRs?
- Is there an easier way to do it?
- Should some corner cases be excluded?

Interface and usecase
---------------------

- Is it aligned with other Kokkos interfaces?
- Is the interface of all functions/classes/etc. intuitive? Does it have a descriptive name?
- How is it used in code in the end?

Documentation
-------------

- Does it need documentation? If yes, is it already in a PR?
- Does it need comments in the implementation to express intent?

Interactions
------------

- Does an abstraction/function have opaque dependencies on non-local variables?
- Are interfaces or variables hidden or shadowed?
- Are any non-local variables changed?

Includes
--------

- Does the header use minimal includes?
- Are the new headers self-contained?

Run
---

- Does the CI work?
- Can I run it and try to break it?

Communication
-------------

- Am I using a helpful, neutral tone?
- Am I giving enough info to the author?

Final Questions
---------------

- Does it definitely improve the current state?
- Am I willing to maintain this code/concept/abstraction/idea in the future?
