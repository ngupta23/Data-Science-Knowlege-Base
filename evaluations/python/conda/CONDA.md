* Check out [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details

# Creating a New Environment

```
conda create -n <myenv> python=3.6
```


# Creating a Conda environment from a requirements.txt file

Click [here](https://stackoverflow.com/questions/48787250/set-up-virtualenv-using-a-requirements-txt-generated-by-conda) for more information

## Creating the environment

**Note that there is a sample requirements.txt file in this folder which can be used for creating future environments**

```
conda create --name <env_name> --file requirements.txt
```

## Exporting the requirements.txt for an environment

From an existing environemnt, run the following command

```
pip freeze > requirements.txt
```

# Updating all packages in a conda environment
```
conda update --all
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


# Deleting an existing environment
```
conda env remove -n <myenv>
```

