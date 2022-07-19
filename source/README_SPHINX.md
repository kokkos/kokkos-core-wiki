# Kokkos Documentation Workflow -- Transitioning to Sphinx<br/><br/>

The `kokkos-core-wiki` is the canonical source of Kokkos documentation "Truth".<br/>

Please do not directly edit the main Kokkos project Wiki.<br/>

Sandia: Creation of wholly new pages will require Review and Approval (R&A).  Check with your Sandia team lead if you have questions about the R&A process.<br/><br/>

* Step 1) FORK:  Fork the kokkos-core-wiki repo (`https://github.com/kokkos/kokkos-core-wiki`) if you do not yet have your own fork.<br/>

* Step 2) CLONE and ADD REMOTES:  Clone your fork and add upstream `kokkos-core-wiki` repo as a remote (`git clone git@github.com:ajpowelsnl/kokkos-core-wiki.git`, `git remote add upstream git@github.com:kokkos/kokkos-core-wiki.git`) if you have not already done so. If you already have a fork of the `kokkos-core-wiki`, then simply update your fork and checkout the branch we'll use for transitioning the documentation (`git fetch upstream -p`, `git checkout upstream/sphinx_porting_stage1`).<br/>

* Step 3) CHECK REMOTES:  Check your remotes; you should see both your fork (`origin`) and the main kokkos-core-wiki project repo (`upstream`):<br/>

```
[ajpowel@kokkos-dev-2 kokkos-core-wiki]$ git remote -v
origin git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
origin git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)
upstream https://github.com/kokkos/kokkos-core-wiki.git (fetch)
upstream https://github.com/kokkos/kokkos-core-wiki.git (push)
```

* Step 4) CREATE A TOPIC BRANCH:  Create a topic branch in your local fork of the kokkos-core-wiki:<br/>

```
git checkout -b documentation/my_topic_branch
```

* Step 5) INSTALL PYTHON MODULES:  For the Sphinx-based workflow, please install the Python modules (using `pip` or `conda`) below on your local machine.  You will only need to install these modules once. <br/>

```
Sphinx
furo
myst-parser
sphinx-copybutton
m2r2
```

* Step 6) BUILD SPHINX DOCUMENTATION:  Build the Sphinx documentation, and check the rendering of the local `html/index.html` file in a browser (suggested: Google Chrome, Firefox):<br/>

```
make html
```

* Step 7) ADD REMOTE FOR PREVIEWING:  Add your fork's  *Kokkos project Wiki* repo as a remote.  Push changes to this repo to preview before creating a pull request (on `sphinx_porting_stage1`).  If your *Kokkos project Wiki* is empty, then you will need to create a first page.  To create this first page, navigate to the Wiki in the banner (*e.g.*, `https://github.com/ajpowelsnl/kokkos/wiki`), and click the green button that says, `Create the first page`.  Once you have created this first page, you should now be able to add the Wiki repo as a remote:<br/>

```
git remote add my_wiki_preview git@github.com:ajpowelsnl/kokkos.wiki.git
```


* Step 8) REBASE TOPIC BRANCH:  Fetch upstream changes, and rebase on your topic branch (`documentation/my_topic_branch`):<br/>

```
git checkout upstream/main
git fetch upstream main
git checkout documentation/my_topic_branch
git rebase upstream/main
```

* Step 9) PREVIEW PROPOSED CHANGES:  Make the desired changes (on your local topic branch), and push to your fork of the main project Wiki (`https://github.com/ajpowelsnl/kokkos/wiki`). During the transition from the github Wiki to Sphnix, you will modify files in the `source` directory, and create pull requests on the `sphinx_porting_stage1` branch.  *Nota bene*: for your first commit, you will need to use the `-f` option to push; this option will overwrite existing files.<br/>

```
git push -f my_wiki_preview documentation/my_topic_branch:master
```

* Step 10) CREATE PULL REQUEST:  If your previewed changes are good, push your local topic branch to your fork of the `kokkos-core-wiki` repo to create a pull request on the `sphinx_porting_stage1` branch.  Please remember that you cannot push directly to the `main` branch of the remote repo:<br/>

```
git push -f origin documentation/my_topic_branch 
```
