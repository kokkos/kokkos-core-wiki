# kokkos-core-documentation-website
Welcome to the Kokkos Documentation repository.  This is the source for https://kokkos.github.io/kokkos-core-wiki/.

## Requirements to build html page locally

The documentation requires Python 3.12 to build. Usually this can be installed with a system package manager, but if your system does not support python 3.12 you can install it easily with [pyenv](https://github.com/pyenv/pyenv).
This is needed just for local render of documentation, so it can be checked before push.
Requirements are in `build_requirements.txt`
Could be installed with: `pip install -r build_requirements.txt`

We recommend using a virtual environment, e.g. if your system python is >= 3.12:

```sh
python -m venv venv
source venv/bin/activate
pip install -r build_requirements.txt
```

Or if you are using pyenv:

```sh
pyenv install 3.12
pyenv shell 3.12
python -m venv venv
pyenv shell --unset
source venv/bin/activate
pip install -r build_requirements.txt
```

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
