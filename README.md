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

## Displaying the site locally

`docs/generated_docs/index.html` can be opened in a web browser, or alternatively you can use python's built-in http server:

```bash
cd docs/generated_docs
python3 -m http.server
```

Then, navigate to http://localhost:8000

Alternatively, if you would like to auto refresh every time you run make, the documentation works with [httpwatcher](https://pypi.org/project/httpwatcher/).
