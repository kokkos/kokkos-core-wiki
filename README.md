# kokkos-core-documentation-website
Welcome to the Kokkos Documentation repository.  This is the source for https://kokkos.github.io/kokkos-core-wiki/.

## Requirements to build html page locally

This is needed just for local render of documentation, so it can be checked before push.
Requirements are in `build_requirements.txt`
Could be installed with: `pip install -r build_requirements.txt`

### Build

```
cd docs
make html
```

To clean:
```
cd docs
make clean
```

### Displaying the site locally

`docs/generated_docs/index.html` can be opened in a web browser, or alternatively you can use python's built-in http server:

```bash
cd docs/generated_docs
python3 -m http.server
```

Then, navigate to http://localhost:8000

Alternatively, if you would like to auto refresh every time you run make, the documentation works with [httpwatcher](https://pypi.org/project/httpwatcher/).

## Requirements to build html page in virtual environments

```bash
# 1. install virtualenv (if needed)
python3 -m pip install --user virtualenv

# 2. clone this repository
git clone git@github.com:kokkos/kokkos-core-wiki.git
cd kokkos-core-wiki

# 3. create virtual environment
python3 -m venv env

# 4. activate virtual environment
source env/bin/activate

# 5. install doc requirements
python3 -m pip install -r build_requirements.txt

# 6. build
cd docs
make html

# 7. display
cd docs/generated_docs
python3 -m http.server
```

Usefull commands:
```bash
# list installed packages
python3 -m pip list

# leave virtual environment
deactivate
```

Source: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
