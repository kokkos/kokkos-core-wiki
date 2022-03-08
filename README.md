# Kokkos Documentation Workflow <br/><br/>

Here, we summarize the steps for setting up the kokkos-core-wiki repo, and a
branch-based workflow for updating documentation in the main Kokkos project.  The kokkos-core-wiki is the
canonical source of Kokkos documentation "Truth".  It is no longer possible to
directly edit the main Kokkos project Wiki.<br/>

Sandia employees: Creation of wholly new pages will require Review and Approval (R&A).  Check with your Sandia team lead, if you have questions about the R&A process.<br/><br/>

* **Step 1) Create your own fork of the kokkos-core-wiki repo by navigating to `https://github.com/kokkos/kokkos-core-wiki`, and clicking "Fork" in the upper right of the github page**.<br/>

* **Step 2) Clone the remote kokkos-core-wiki repo, and add your fork as a remote:**<br/>

```
git clone git@github.com:kokkos/kokkos-core-wiki.git
git remote add ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git
```

* **Step 3) Check your remotes; you should see both your fork (`ajpowelsnl`) and the main kokkos-core-wiki project repo (`origin`):**<br/>

```
[ajpowel@kokkos-dev-2 kokkos-core-wiki]$ git remote -v
ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)
origin https://github.com/kokkos/kokkos-core-wiki.git (fetch)
origin https://github.com/kokkos/kokkos-core-wiki.git (push)
```

* **Step 4) Create a topic branch in your local fork of the kokkos-core-wiki:**<br/>

```
git checkout -b demo/setup
```

* **Step 5) Add your fork's  *Kokkos project Wiki* repo as a remote (you will push changes to this repo to preview before creating a pull request on `origin git@github.com:kokkos/kokkos-core-wiki.git`).  If your *Kokkos project Wiki* is empty, then you will need to create a first page.  To create this first page, navigate to the Wiki in the banner (*e.g.*, `https://github.com/ajpowelsnl/kokkos1-kernels/wiki`), and click the green button that says, `Create the first page`.  Once you have created this first page, you should now be able to add the Wiki repo as a remote:**<br/>

```
git remote add my_wiki git@github.com:ajpowelsnl/kokkos.wiki.git
```

* **Step 6) Verify your remotes.  You should now have three different repos: your local, forked kokkos-core-wiki (`ajpowelsnl`), the remote kokkos-core-wiki (`origin`), and the Wiki associated with your forked Kokkos project (`my_wiki`):**<br/>

```
[ajpowel@kokkos-dev-2 kokkos-core-wiki]$ git remote -v
ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)
my_wiki git@github.com:ajpowelsnl/kokkos.wiki.git (fetch)
my_wiki git@github.com:ajpowelsnl/kokkos.wiki.git (push)
origin https://github.com/kokkos/kokkos-core-wiki.git (fetch)
origin https://github.com/kokkos/kokkos-core-wiki.git (push)
```

* **Step 7) Fetch new changes to the remote (`origin`), and rebase on your topic branch (`demo/setup`) to update:**<br/>

```
git fetch origin main
git checkout demo/setup
git rebase origin/main
```

* **Step 8) Make the desired changes (on your local topic branch), and push to the main project Wiki (of your fork).  Nota bene: for your first commit, you will need to use the `-f` option to push; this option will overwrite existing files.**<br/>

```
git push -f my_wiki demo/setup:master
```

* **Step 9) Preview your changes by navigating to the main project Wiki of your fork:**<br/>

```
https://github.com/ajpowelsnl/kokkos/wiki
```

* **Step 10) If your previewed changes are good, push your local topic branch to your fork of the kokkos-core-wiki repo to create a pull request.  You cannot push directly to the `main` branch of the remote repo:**<br/>

```
git push -f ajpowelsnl demo/setup 
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
