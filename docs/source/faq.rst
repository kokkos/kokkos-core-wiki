FAQ
###

.. _join-slack-workspace:

**How do I join the Kokkos slack channel?**
  You can find the slack channel at `kokkosteam.slack.com <https://kokkosteam.slack.com>`_. Register a new account with your email. We reached the limit of whitelisted organizations, but every member of the Kokkos Slack workspace can invite more people. If no one you know is in the Slack workspace you can contact the Kokkos maintainers (their emails are in the LICENSE file).

**How do I compile Kokkos with C++20 or C++23?**
  When configuring Kokkos with cmake, add the flag ``-DCMAKE_CXX_STANDARD=20`` (or ``23``). Ensure that the flag is also set for any downstream applications.
