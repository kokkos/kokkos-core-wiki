# Kokkos Testing Processes and Change Process

Kokkos testing falls into three categories:

 - Pull Request Testing
 - Nightly Testing
 - Integration Testing (Release Testing)
 
## Pull Request Testing

All changes to Kokkos are introduced via pull requests against the github.com develop branch of Kokkos. 
In order to be merged two conditions must be met:

1) Automatic testing of the pull request must pass.
2) A Kokkos core developer must approve the pull request, after checking the changes for alignment with Kokkos developer standards. 

The tested configurations in Pull Request testing cover the major deployment systems
and are executed via jenkins and travis at various institutions. 
Pull request testing also includes verification that the formatting meets 
the clang-format style specified in the repository. 
Test configurations are defined in the `kokkos/.jenkins` and `kokkos/.travis.yml` files.
The clang-format style file is `kokkos/.clang-format`.

## Nightly Testing

Nightly testing covers a wider range of compilers and configuration of Kokkos 
on an extensive list of platforms. 
Test configurations are given in `kokkos/scripts/testings/test_all_sandia`. 
Executing this script on the (in it) specified platforms will meet full testing requirements.
Nightly tests are set up via Jenkins and execute this script in stages. 

## Integration Testing (Release Testing)

In order for a new Kokkos version to be released full integration testing is performed.
A release is then formed by merging the Kokkos develop branch into its master branch, 
and creating a git tag with the version number. 
Details of the process are described in Testing Process Details. 

