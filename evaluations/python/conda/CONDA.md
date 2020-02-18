* Check out [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details

# Creating a New Environment

```
conda create -n myenv python=3.6
```

# Displaying existing environments

```
conda env list
```

# Renaming Conda env in jupyter
Click [here](https://stackoverflow.com/questions/39604271/conda-environments-not-showing-up-in-jupyter-notebook) for more information

```
source activate <myenv>
python -m ipykernel install --user --name <myenv> --display-name "<display_name>"
```
