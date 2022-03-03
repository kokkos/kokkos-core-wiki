########################################################
#			Welcome to the Kokkos Documentation Workflow     #
########################################################

	Here, we summarize the steps for setting up the kokkos-core-wiki repo, and a
branch-based workflow for updating documentation in the main Kokkos project.  The kokkos-core-wiki is the
canonical source of Kokkos documentation "Truth".

	CAVEAT: Creation of wholly new pages will require Review and Approval in some
cases.  Check with your team lead, if you are uncertain.

	STEP 1) Create your own fork of the kokkos-core-wiki repo by navigating to
					https://github.com/kokkos/kokkos-core-wiki, and clicking "Fork" in the upper
					right of the github page.

	STEP 2) Clone your forked repo:

					git clone git@github.com:ajpowelsnl/kokkos-core-wiki.git

	STEP 3) Create a topic branch in your local fork of the kokkos-core-wiki:

					git checkout -b demo/setup

	STEP 4) Check your remotes:

					[ajpowel@kokkos-dev-2 kokkos-core-wiki]$ git remote -v
					origin	git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
					origin	git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)

	STEP 5) Rename your (local) repo to disambiguate:

					git remote rename origin ajpowelsnl

	STEP 6) Add the remote kokkos-core-wiki repo (i.e., the repo you will create a PR on):

					git remote add origin https://github.com/kokkos/kokkos-core-wiki.git

	STEP 7) Add the Kokkos project Wiki from your fork (you will push changes to this repo to preview): 

 					git remote add my_wiki git@github.com:ajpowelsnl/kokkos.wiki.git

	STEP 8) Verify your remotes (you should now have three different repos: 
					your local, forked kokkos-core-wiki ("ajpowelsnl" below), the remote kokkos-core-wiki ("origin" below),
					and a local main-project Wiki associated with your forked Kokkos project ("my_wiki" below)):


				git remote -v
	
				ajpowelsnl	git@github.com:ajpowelsnl/kokkos-core-wiki.git (fetch)
				ajpowelsnl	git@github.com:ajpowelsnl/kokkos-core-wiki.git (push)
				my_wiki	git@github.com:ajpowelsnl/kokkos.wiki.git (fetch)
				my_wiki	git@github.com:ajpowelsnl/kokkos.wiki.git (push)
				origin	https://github.com/kokkos/kokkos-core-wiki.git (fetch)
				origin	https://github.com/kokkos/kokkos-core-wiki.git (push)
 
	STEP 9) Set up your local topic branch track origin/main:

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

	STEP 10) Update your topic branch,rebasing on origin/main:
				
				git checkout demo/setup
				git rebase main


	STEP 10) Make the desired changes (on your local topic branch), and push
				those changes to the main project Wiki (of your fork).  Nota bene: you
				will need to use the "-f" option for the **FIRST** push; this option will overwrite the file.  

				git push -f my_wiki demo/setup:master



	STEP 11) Preview your changes by navigating to the main project Wiki of your
				fork:

				https://github.com/ajpowelsnl/kokkos/wiki

	STEP 12) If your previewed changes are good, push your local topic branch to
				your fork of the kokkos-core-wiki to create a pull request.  Please
				note that pushes to the main branch will be automatically, immediately
				deployed to the Kokkos Wiki:

				git push ajpowelsnl demo/setup 
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
