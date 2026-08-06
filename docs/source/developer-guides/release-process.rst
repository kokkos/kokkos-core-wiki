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

