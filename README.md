# kokkos-core-documentation project
Welcome to the Kokkos Documentation repository.  This is the source for https://kokkos.github.io/kokkos-core-wiki/.

## Requirements to build html page locally

The library-level dependencies needed for Kokkos documentation are:

- Sphinx
- furo
- myst-parser
- sphinx-copybutton
- m2r2

Each of the above libraries can be installed using `conda` or `pip`. 

These libraries are needed for local documentation rendering, a critical check *before* creating pull requests (PR), and updating existing PR.

These requirements are in `build_requirements.txt,` and can be installed with: `pip install -r build_requirements.txt`

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

```
bash
cd docs/generated_docs
python3 -m http.server
```

Then, navigate to http://localhost:8000

- For `conda` or `pip`, please install `pytest-httpserver`

Alternatively, if you would like to auto refresh every time you run make, the documentation works with [httpwatcher](https://pypi.org/project/httpwatcher/).  This library is only available for `pip` installation.
