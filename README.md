# Discovering Patterns in PISA


## Semester Master Project in Data Science

- **Author :** Ajkuna Seipi
- **Lab :** ML4ED
- **Semester :** Spring 2024

### Abstract

At the ML4ED laboratory at EPFL, in collaboration with the HEP of Lausanne, we analyzed students' interactions with the RobotArm programming game from PISA 2025 to explore their reasoning patterns in three learning tasks.
Our study emphasized analyzing the process of block-based programming rather than just the outcomes, as focusing solely on output covers different reasoning and learning trajectories. We introduced a distance metric for the game’s world space to understand spatial relationships and movements, comparing it to the code space to link programming actions with in-game effects.

### Data

The data used are in csv format. 
Two datasets where used: 
* `data_code_space.csv` : csv file containing the pre-processed XML files, and the computed successive distances of the code space, e.g Tree Edit Distance of ASTs. 
* `ldw_2023_pilot_coding_tasks_outputs_processed_3.csv` : csv file containing the world space representation for each student for each attempt, and their score, e.g. if they succeeded or not. 

The datasets should be inside the `data` folder, in the root of the project. 

### Getting Started 

The script to install the requirements for the project: 
```bash
# Install the required packages
pip install -r requirements.txt
```

The project is divided into four parts: 
1. Evaluation of a distance metric for the world space. 
2. Students' data preprocessing and qualitative analysis.
2. Pattern mining on the different spaces. 
3. Predicitve models for correlation and success. 

#### Evaluation of a distance metric 

Several distances were considered, and their implementation is in the `distances.py` file inside the `src` folder. 

Some tests were generated to evaluate the effectiveness of the distances. 
They can be run with the following command: 
```bash
cd src
python run_distance_tests.py
```
The command will print some sanity check tests and will plot some graphs used to compare the distances. 
The different tests are explained in the following link: [Test cases](https://docs.google.com/presentation/d/1Y8Axf3NsCZMb9q7qWSoVY0sjUH5uFNKZ-j03dZ0UJLs/edit?usp=sharing)

* `metric_tests.py` : contains the generated tests for evaluating the different metrics for the world distance.

### Students' data preprocessing 

Inside the `src/` folder, you can find the following files: 
* `preprocess.py` : contains the functions to preprocess the whole dataset of Student's attempts. 
* `plot_analysis.py` : contains all the plot functions to analysis the Students' attempts (world distacances, code distances, errors etc.)

To preprocess the dataset and generate all the plots for a qualitative analysis, you can run the following command inside the `src/` folder: 
```bash
python run_preprocessing.py
```
### Sequential Pattern Mining: 

To perform SPM, we used the code of [Kinnebrew, J. S. and al. (2013)](https://files.eric.ed.gov/fulltext/EJ1115377.pdf). 

The SPM algorithm consists of two main steps:
1. Computes the student support for each possible pattern. 
2. Computes the interaction support for each pattern for each student.

You need to go inside the folder `pattern_mining/` in `src/` to find all the experiments generated with the SPM algorithm on our dataset. 

The folder `experiments/` contains several experiments, which mainly consist of the pattern mining done on the **error sequences**, **case sequences**, **successive world distance sequences** and **successive code distance sequences**. 

The folder `plots/` contains the heatmaps of the experiments, with the frequent patterns, their *s-support* and their *i-support*. 

The file `plots_pattern.py` contains the functions to plot the patterns in a heatmap. 

To generate all the plots of the experiments, you need to run this command inside the `pattern_mining` folder: 
```bash
python generate_plots.py
```
A folder `plots/` should be inside the `pattern_mining/` folder, so that the heatmaps are saved inside. 

### Predictive models

Finally, to access to the prediction files and experiments, you need to navigate inside the `prediction/` folder in `src/`. 

You can find a description of each file below: 
* `utils.py`: contains some utility functions used inside the main training and preprocessing files. 
* `models.py`: contains the models used for the prediction, implemented with pytorch. 
* `sequence_creator.py`: contains the class to preprocess the students' attempts into sequences. 
* `model_trainer.py`: contains the class to perform training and cross validation. 
* `experiment_runner.py`: contains the main function to run the training or cross-validation on defined experiments, e.g. specific model, task, split_type, hyper-parameters, etc. 
* `run_experiments.sh`: is a bash script to perform the different experiments on different data files. 

To run the experiments, you need to run the following command inside the `prediction/` folder: 
```bash
./run_experiments.sh
```

The folder `results/` contains the results of our experiments in json format. 

A folder `plots/` should be inside the `prediction/` folder, so that the plots generated by the experiments are saved inside. 

### Other

Throughout the analysis, we used several notebooks, which can be found inside the `notebooks/` folder in the root of the project. 

The `plots/` folder in the root of the projet contains the plots of the distance metrics tests and the plots for the qualitative analysis of the students' attempts.



