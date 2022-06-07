# Kokkos Documentation Workflow <br/><br/>

The kokkos-core-wiki is the canonical source of Kokkos documentation "Truth".  Please do not directly edit the main Kokkos project Wiki.<br/>

Sandia employees: Creation of wholly new pages will require Review and Approval (R&A).  Check with your Sandia team lead if you have questions about the R&A process.<br/><br/>

* Step 1) Fork the kokkos-core-wiki repo (`https://github.com/kokkos/kokkos-core-wiki`) if you do not yet have your own fork.<br/>

* Step 2) Clone your fork and add upstream `kokkos-core-wiki` repo as a remote (`git clone git@github.com:ajpowelsnl/kokkos-core-wiki.git`, `git remote add upstream git@github.com:kokkos/kokkos-core-wiki.git`) if you have not already done so. If you already have a fork of the `kokkos-core-wiki`, then simply update your fork and checkout the branch we'll use for transitioning the documentation (`git fetch upstream -p`, `git checkout upstream/sphinx_porting_stage1`).<br/>

* Step 3) Check your remotes; you should see both your fork (`origin`) and the main kokkos-core-wiki project repo (`upstream`):<br/>

```
[ajpowel@kokkos-dev-2 kokkos-core-wiki]$ git remote -v
origin git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
origin git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)
upstream https://github.com/kokkos/kokkos-core-wiki.git (fetch)
upstream https://github.com/kokkos/kokkos-core-wiki.git (push)
```

* Step 4) Create a topic branch in your local fork of the kokkos-core-wiki:<br/>

```
git checkout -b documentation/my_topic_branch
```

* Step 5) For the Sphinx-based workflow, please install the Python modules (using `pip` or `conda`) below on your local machine.  You will only need to install these modules once. <br/>

```
Sphinx
furo
myst-parser
sphinx-copybutton
m2r2
```

* Step 6) Build the Sphinx documentation, and check the rendering of the local `html/index.html` file in a browser (suggested: Google Chrome, Firefox):<br/>

```
make html
```

* Step 7) Add your fork's  *Kokkos project Wiki* repo as a remote (you will push changes to this repo to preview before creating a pull request on `upstream git@github.com:kokkos/kokkos-core-wiki.git`).  If your *Kokkos project Wiki* is empty, then you will need to create a first page.  To create this first page, navigate to the Wiki in the banner (*e.g.*, `https://github.com/ajpowelsnl/kokkos/wiki`), and click the green button that says, `Create the first page`.  Once you have created this first page, you should now be able to add the Wiki repo as a remote:<br/>

```
git remote add my_wiki_preview git@github.com:ajpowelsnl/kokkos.wiki.git
```

* Step 8) Verify your remotes.  You should now have three different repos: your local, forked kokkos-core-wiki (`ajpowelsnl`), the remote kokkos-core-wiki (`upstream`), and the Wiki associated with your forked Kokkos project (`my_wiki_preview`):<br/>

```
[ajpowel@kokkos-dev-2 kokkos-core-wiki]$ git remote -v
origin git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
origin git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)
my_wiki_preview https://github.com:ajpowelsnl/kokkos.wiki.git (fetch)
my_wiki_preview https://github.com:ajpowelsnl/kokkos.wiki.git (push)
upstream git@github.com/kokkos/kokkos-core-wiki.git (fetch)
upstream git@github.com/kokkos/kokkos-core-wiki.git (push)
```

* Step 9) Fetch new changes from the remote kokkos-core-wiki (`upstream`), and rebase on your topic branch (`documentation/my_topic_branch`) to update:<br/>

```
git checkout upstream/main
git fetch upstream main
git checkout documentation/my_topic_branch
git rebase upstream/main
```

* Step 10) Make the desired changes (on your local topic branch), and push to the main project Wiki (of your fork). During the transition from the github Wiki to Sphnix-based documentation, you will modify files in the `source` directory, and create pull requests on the `sphinx_porting_stage1` branch.  *Nota bene*: for your first commit, you will need to use the `-f` option to push; this option will overwrite existing files.<br/>

```
git push -f my_wiki_preview documentation/my_topic_branch:master
```

* Step 11) Preview your changes by navigating to the main project Wiki of your fork:<br/>

```
https://github.com/ajpowelsnl/kokkos/wiki
```

* Step 12) If your previewed changes are good, push your local topic branch to your fork of the kokkos-core-wiki repo to create a pull request.  You cannot push directly to the `main` branch of the remote repo:<br/>

```
git push -f origin demo/setup 
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Delta compression using up to 40 threads
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 305 bytes | 305.00 KiB/s, done.
Total 3 (delta 2), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (2/2), completed with 2 local objects.
remote: 
remote: Create a pull request for 'demo/setup' on GitHub by visiting:
remote:      https://github.com/ajpowelsnl/kokkos-core-wiki/pull/new/demo/setup
remote: 
To github.com:ajpowelsnl/kokkos-core-wiki.git
* [new branch]      demo/setup -> demo/setup
```
