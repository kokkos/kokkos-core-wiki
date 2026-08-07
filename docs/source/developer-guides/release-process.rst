Release Process
===============

(Work In Progress)

Create Release Branch and Update Project Version
-------------------------------------------------
(For maintainers)

.. note::
   **Version Numbering Scheme**

   Development versions use ``.99`` as the patch number to ensure they always
   compare as more recent than any release in that series (e.g., ``4.2.99`` >
   ``4.2.5``). This allows code to reliably discriminate between development
   and release versions.

Feature Releases (X.Y.0)
~~~~~~~~~~~~~~~~~~~~~~~~

.. important::
   Steps 1-7 must be completed in sequence without merging other changes to
   ``develop``. This ensures accurate version tracking throughout the codebase.

1. Check that the ``develop`` branch is in decent shape with the Continuous
   Integration Working Group, that all nightly builds are passing, and that there
   are no outstanding issues in the integration testing.

2. Create the release candidate branch:

.. code-block:: console

   git checkout -b release-candidate-X.Y.0

3. Update the version number to ``X.Y.0`` in the root ``CMakeLists.txt``:

.. code-block:: cmake

   # Edit these lines in CMakeLists.txt:
   set(Kokkos_VERSION_MAJOR X)
   set(Kokkos_VERSION_MINOR Y)
   set(Kokkos_VERSION_PATCH 0)

Then commit the change:

.. code-block:: console

   git commit -s -m 'Set version number to X.Y.0' CMakeLists.txt

4. Push the release candidate branch to the upstream repository:

.. code-block:: console

   git push https://github.com/kokkos/kokkos.git release-candidate-X.Y.0

5. Create and checkout a new ``bump_version_number`` branch with ``develop`` as
   the starting point:

.. code-block:: console

   git checkout -b bump_version_number develop

6. Update the version from ``X.(Y-1).99`` to ``X.Y.99`` in the root ``CMakeLists.txt``:

.. code-block:: cmake

   # Edit these lines in CMakeLists.txt:
   set(Kokkos_VERSION_MAJOR X)
   set(Kokkos_VERSION_MINOR Y)
   set(Kokkos_VERSION_PATCH 99)

Then commit the change:

.. code-block:: console

   git commit -s -m 'Bump version from X.(Y-1).99 to X.Y.99' CMakeLists.txt

7. Push to your fork and open a pull request against the ``develop`` branch:

.. code-block:: console

   git push <your-fork-remote> bump_version_number

**This pull request must be merged immediately before any other feature PRs**
to maintain version integrity in the development branch.

8. Open a tracker issue for the ``X.(Y+1)`` changelog and pin it to the
   repository (use GitHub's "Pin issue" feature). Unpin the old ``X.Y``
   changelog issue. Use the old issue as a template, keeping all sections
   but clearing the entries.

9. Notify developers on the `#nucleus
   <https://kokkosteam.slack.com/archives/G5CBLMFLP>`_ channel that the release
   branch has been created and that the version bump PR needs to be merged as
   the next change to ``develop``.

Patch Releases (X.Y.Z)
~~~~~~~~~~~~~~~~~~~~~~

Patch releases are created from the previous release tag to incorporate
critical bug fixes into an existing release series.


.. note::
   Unlike feature releases, patch releases do not require updating the
   ``develop`` branch version, as it already uses the ``.99`` patch number for
   the current development series.

1. Create the release candidate branch from the latest patch release tag:

.. code-block:: console

   git checkout -b release-candidate-X.Y.(Z+1) X.Y.Z

2. Update the version number from ``X.Y.Z`` to ``X.Y.(Z+1)`` in the root ``CMakeLists.txt``:

.. code-block:: cmake

   # Edit these lines in CMakeLists.txt:
   set(Kokkos_VERSION_MAJOR X)
   set(Kokkos_VERSION_MINOR Y)
   set(Kokkos_VERSION_PATCH Z+1)

Then commit the change:

.. code-block:: console

   git commit -s -m 'Bump version from X.Y.Z to X.Y.(Z+1)' CMakeLists.txt

3. Push the release candidate branch to the upstream repository:

.. code-block:: console

   git push https://github.com/kokkos/kokkos.git release-candidate-X.Y.(Z+1)

4. Proceed to cherry-picking approved changes (see next section).


Cherry-Picking Changes into Release Candidates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**General Workflow**

Changes should follow the develop-first workflow unless there is an exceptional
reason, such as:

- Develop has diverged significantly, making cherry-picking impractical
- Develop is temporarily broken or in an untestable state
- The bug only exists in the release branch due to subsequent fixes in develop
- Develop contains incompatible changes requiring a fundamentally different fix

1. **Merge to develop first:** All changes must be integrated and tested on the
   ``develop`` branch before being considered for backporting.

2. **Get authorization:** Once merged to develop, obtain approval from a maintainer,
   through the weekly developer meeting, or on the `#nucleus <https://kokkosteam.slack.com/archives/G5CBLMFLP>`_
   channel before backporting.

3. **Open cherry-pick PR:** Create a pull request targeting the release candidate
   branch with:

   - **Title format:** ``[X.Y.Z] Original Well-Crafted Subject Line``
   - **Description starts with:** "Cherry-picking the changes from PR #1234 into
     the X.Y.(Z+1) release candidate branch"

.. tip::
   Developers are encouraged (but not required) to seek approval **before**
   opening the backport PR to avoid unnecessary work if the change is deemed
   inappropriate for the release.

**Scope Guidelines for Feature Release Candidates (X.Y.0)**

During the release candidate phase for a new feature release, patches should be
limited to:

- **Bug fixes** discovered during testing
- **Important optimization improvements** that significantly impact performance
- **Completion of features** that were started before the branch was created

.. warning::
   **As the release date approaches**, patches should be increasingly conservative
   and limited to:

   - Critical bugs that affect core functionality
   - Regressions from the previous release
   - Build system failures on supported platforms

**Scope Guidelines for Patch Releases (X.Y.Z, Z > 0)**

Patches for bug fix releases have stricter requirements:

- **Bug fixes only** (preferred)
- **Very safe and critical performance improvements** (requires strong justification)
- **Must maintain full API compatibility** with the X.Y.0 release

.. important::
   Patch releases exist to provide stability for users who have already deployed
   the X.Y.0 release. Breaking changes of any kind are not acceptable.


Final Tasks
-----------
(For maintainers)

**Automated Steps**

1. Tag and push the release:

.. code-block:: console

    git tag --sign X.Y.Z
    git push https://github.com/kokkos/kokkos.git X.Y.Z

The ``release`` workflow will automatically generate downloadable source code
archives (``.zip`` and ``.tar.gz``) for the ``X.Y.Z`` Git tag, it will compute
the corresponding SHA-256 checksums, create a ``kokkos-X.Y.Z-SHA-256.txt``
file, and upload them to the release page.
It will draft release notes with tables gathering links to the source
distributions and summary files.

**Manual Verification Steps**

2. Verify checksums:

Once the GitHub CI Action has run, download and verify the integrity of the
generated artifacts:


.. code-block:: console

  sha256sum -c kokkos-X.Y.Z-SHA-256.txt

3. Sign the checksum file:

.. code-block:: console

  gpg --detach-sig --armor kokkos-X.Y.Z-SHA-256.txt

**Publish the Release**

4. Navigate to the release page on GitHub
(https://github.com/kokkos/kokkos/releases/latest):

- Click the "Edit" button
- Adjust the release date to ``YYYY-MM-DD``
- In the release notes, add your GPG signing key information (e.g.,
  ``Digitally signed with [Key short ID](link-to-public-key)``)
- Upload the ``kokkos-X.Y.Z-SHA-256.txt.asc`` signature file
- Click "Update release"

Double-check that everything looks good. Congratulations, the release is
shipped!

.. note::
   The CI/CD Action publishes the ``X.Y.Z`` release as the "Latest" release on
   GitHub. If this is a patch release for an older version (e.g., releasing
   4.7.3 after 5.0.0 is already out), you may need to manually change which
   release is marked as "Latest".

