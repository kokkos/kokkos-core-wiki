Release Process
===============

(Work In Progress)

Final Tasks
-----------
(For maintainers)

**Prerequisites**

- Write access to the Kokkos repository
- GPG key configured for signing
- Git configured to sign tags

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

Once the GitHub CI Action has run, verify the integrity of the generated
artifacts:


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

