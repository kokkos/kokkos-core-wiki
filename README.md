# kokkos-core-documentation-website
Welcome to the Kokkos Documentation repository.  This is the source for https://kokkos.github.io/kokkos-core-wiki/.

## Requirements to build html page locally

This is needed just for local render of documentation, so it can be checked before push.
Requirements are in `build_requirements.txt`
Could be installed with: `pip install -r build_requirements.txt`

## Build

```
cd docs
make html
```

To clean:
```
cd docs
make clean
```
