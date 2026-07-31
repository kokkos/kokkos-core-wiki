Contributing
============

.. toctree::
   :maxdepth: 1
   :hidden:

   templates/index
   developer-guides/index
   testing-and-issue-tracking

We welcome external contributions. Please `open an issue
<https://github.com/kokkos/kokkos/issues>`_ to discuss your changes
first—especially for larger features—and submit your pull request against the
``develop`` branch.
If you are unsure about opening an issue, feel free to `reach out on Slack
<https://kokkos.org/community/chat/#slack>`__ for initial feedback.

Legal Requirements
------------------
License
^^^^^^^
By contributing to Kokkos Core, you agree that your contribution will be
licensed under the **Apache License 2.0 with LLVM Exception**.  This exception
allows Kokkos to be compiled and linked into binaries (including closed-source
commercial software) without requiring the end-user to distribute the Kokkos
license text. See the `LICENSE <license.html>`__ for details. Contributors (or
their employers) retain copyright on their own contributions.

Developer Certificate of Origin (DCO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To ensure clear chain of custody for open-source software, we require all
contributions to be "signed off" according to the `Developer Certificate of
Origin <https://developercertificate.org/>`_.

By adding a ``Signed-off-by`` line to your commit message, you certify that
you have the right to submit the work under the project's license.

If using the command line, you can automate this by adding the ``-s`` flag:

.. code-block:: bash

   git commit -s -m "My informative commit message"

If you are using a Git GUI or web interface, you must manually type the line at the 
very end of your commit description:

.. code-block:: text

   Signed-off-by: Jane Doe <jane.doe@example.com>

Generative AI and Assisted-By Contributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The use of Generative AI tools (such as GitHub Copilot, ChatGPT, Claude, or
similar models) to assist in writing code for Kokkos is permitted under strict
conditions of human oversight and transparency.

**Human Accountability:**
AI-generated code often contains subtle logic bugs, security risks, or
licensing compliance issues. Because AI tools cannot sign a Developer
Certificate of Origin (DCO), **you** remain fully responsible for every line
of code you submit. You must thoroughly review, test, and understand all
AI-assisted contributions before submitting them.

**Attribution and Disclosure:**
To maintain a clear and transparent history of how code is authored, any
commit that contains a significant amount of code generated or heavily modified
by an AI tool must include an ``Assisted-By`` trailer in the commit message.

This trailer should specify the tool name and version used, placed directly alongside
your ``Signed-off-by`` line.

.. code-block:: text

   Fix out-of-bounds error in View layout initialization

   Detailed explanation of the fix goes here...

   Assisted-By: Copilot:gpt-4o
   Signed-off-by: Jane Doe <jane.doe@example.com>

Submissions generated entirely by automated AI agents without active human
oversight and code review are strictly prohibited.

Contributing Documentation
--------------------------

Please see the `README <https://github.com/kokkos/kokkos-core-wiki/blob/main/README.md>`_ for general instructions on building the documentation.

To make it easier to contribute API documentation, we have a page of documentation templates :doc:`here <templates/index>`

Developers' Corner
------------------

* :doc:`Developer Guide  <developer-guides/index>`

* :doc:`Kokkos Planning and Testing <testing-and-issue-tracking>`

