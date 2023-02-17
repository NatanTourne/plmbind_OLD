# Landing page

In your environment, navigate to the root folder of this repo and
```bash
pip install -e .
```
Now you have installed anything under "plmbind" as a package and can use it in any folder.

## Project structure / guidelines
- A `README.md` containing thoughts, notes, todos, and where you keep track of things you did in order to make everything reproducible (instructions for how to obtain the exact same data set you obtained, up until the very details of which version of remap, which version of esm, etc .. was used and how they were combined etc)
- `LICENSE` files and `.gitignore` files. A standard gitignore should be enough (the one I included is the github default for python), data folders should not be in the gitignore as they should live outside of your github repo anyway. Same goes for notebooks. This is the general structure I follow:
```
/project
├── github_repo
│   ├── whatever_is_in_here
├── playground
│   ├── random_stuff.ipynb
│   ├── figure_making.ipynb
│   ├── tuning_results.ipynb
│   ├── idea_no5842.ipynb
├── data
│   ├── datafile.h5t
│   ├── anything_else
├── logs
│   ├── modelcheckpoint1
│   ├── modelcheckpoint2
```
This structure should make sure everything is organized and no files are contaminated to where they should not belong. As the project is structured as a python package and is install in [editable/development mode](https://letmegooglethat.com/?q=python+development+install), the exact location of the github repository folder relative to all other files is of no importance.

## Todos

- [ ] Refactor the project as a python package
- [ ] Add a section in this README that contains instructions on how to reproduce the data set.