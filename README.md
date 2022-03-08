# Kokkos Documentation Workflow <br/><br/>

Here, we summarize the steps for setting up the kokkos-core-wiki repo, and a
branch-based workflow for updating documentation in the main Kokkos project.  The kokkos-core-wiki is the
canonical source of Kokkos documentation "Truth".<br/>

Sandia employees: Creation of wholly new pages will require Review and Approval (R&A).  Check with your Sandia team lead, if you have questions about the R&A process.<br/><br/>

* **STEP 1) Create your own fork of the kokkos-core-wiki repo by navigating to `https://github.com/kokkos/kokkos-core-wiki`, and clicking "Fork" in the upper right of the github page**.<br/>

* **STEP 2) Clone your forked repo, and add it as a remote:**<br/>

```
git clone git@github.com:ajpowelsnl/kokkos-core-wiki.git
git remote add ajpowelsnl
git@github.com:ajpowelsnl/kokkos-core-wiki.git
```

* **STEP 3) Add the remote repo as "origin" or "upstream", according to your preference.  You will create pull requests against this repo:**<br/>

```
git remote add origin git@github.com:kokkos/kokkos-core-wiki.git
```

* **STEP 4) Check your remotes; you should see both your fork (`ajpowelsnl`) and the main project repo (`origin`):**<br/>

```
[ajpowel@kokkos-dev-2 kokkos-core-wiki]$ git remote -v
ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)
origin https://github.com/kokkos/kokkos-core-wiki.git (fetch)
origin https://github.com/kokkos/kokkos-core-wiki.git (push)
```

* **STEP 5) Create a topic branch in your local fork of the kokkos-core-wiki:**<br/>

```
git checkout -b demo/setup
```

* **STEP 6) Add the Kokkos project Wiki repo from your fork as a remote (you will push changes to this repo to preview before creating a pull request on `origin git@github.com:kokkos/kokkos-core-wiki.git`):**<br/>

```
git remote add my_wiki git@github.com:ajpowelsnl/kokkos.wiki.git
```

* **STEP 7) Verify your remotes.  You should now have three different repos: your local, forked kokkos-core-wiki (`ajpowelsnl`), the remote kokkos-core-wiki (`origin`), and the Wiki associated with your forked Kokkos project (`my_wiki`):**<br/>

```
[ajpowel@kokkos-dev-2 kokkos-core-wiki]$ git remote -v
ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
ajpowelsnl git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)
my_wiki git@github.com:ajpowelsnl/kokkos.wiki.git (fetch)
my_wiki git@github.com:ajpowelsnl/kokkos.wiki.git (push)
origin https://github.com/kokkos/kokkos-core-wiki.git (fetch)
origin https://github.com/kokkos/kokkos-core-wiki.git (push)
```

* **STEP 8) Set up your local topic branch to track origin/main:**<br/>

```
git checkout main
git branch --set-upstream-to=origin/main 
git pull origin/main
git show
commit f0e10f59f50f23930a2f4eedfce3eed869344277 (HEAD -> main, origin/main)
Merge: 7b5063e ec9f444
Author: Damien L-G <dalg24@gmail.com>
Date:   Mon Feb 28 21:09:11 2022 -0500

Merge pull request #3 from dalg24/fixup_gh_actions

Fixup path gh workflow
```

* **STEP 9) Update your topic branch, rebasing on origin/main:**<br/>

```
git checkout demo/setup
git rebase main
```

* **STEP 10) Make the desired changes (on your local topic branch), and push to the main project Wiki (of your fork).  Nota bene: you will need to use the `-f` option to push; this option will overwrite existing files.**<br/>

```
git push -f my_wiki demo/setup:master
```

* **STEP 11) Preview your changes by navigating to the main project Wiki of your fork:**<br/>

```
https://github.com/ajpowelsnl/kokkos/wiki
```

* **STEP 12) If your previewed changes are good, push your local topic branch to your fork of the kokkos-core-wiki to create a pull request.  Please note that pushes to your branch will be automatically, immediately deployed to your *fork* of the Kokkos project Wiki.  You cannot push directly to the `main` branch of the remote repo:**<br/>

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
