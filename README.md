# MIENC - Non Linearity in pairwise interactions

Software for systematic testing of the relevance of nonlinearity in bivariate data.

# Installing

Installation is easy, just:
1. (optional but very advised) create and activate a new virtual environment:
    ```bash
    pyhton -m venv mienc
    source mienc/bin/activate
    ```
2. change directory to the unpacked repository:
    ```bash
    cd /path/to/mienc
    ```
1. install the package:
    1. if you only want to use its components for your measures:
        ```bash
        pip install .
    1. if you want to reproduce the results of the paper: 
        ```bash
        pip install .[rep]
        ```
# Running
After the installation you can run `mienc` from command line or access it programmatically more flexible experiments in Python.

Launch the program calling `mienc -h` to know the options, see `data/config_example.ini` to know how to structure the config file that guides the program in accessing and processing data.

The access information for the dataset (at least the path to the file), must be in the config file.

`.mat` files are expected to contain data as [time_samples, series, session].

# Reproducibility
After installing `mienc` with the optional dependencies using `pip install .[rep]` on a powerful enough machine (and with enough disk space: the dataset download will take ~100 GB and the intermediate results about as much) a series of `scripts` will guide the reproduction of the results in the paper.

Edit the file `data/localsettings.ini` adding a path to a folder where all the downloads and results will go.

Run the notebooks in the scrips folder from s01 to s03. You will also need a working Matlab installation for the processing of the fMRI data for the test-retest section and run script s01a when invited to do so in the notebook s01 (after editing a few lines).

Wait.

Done.

# Reading the documentation

If I didn't upload it somewhere else, from the `mip2` folder run:
```bash
pip install -r docs/requirements.txt
```
to get the right packages.

Now you can read the documentation running:
```bash
mkdocs serve
```
And opening your browser at: `http://127.0.0.1:8000/`.