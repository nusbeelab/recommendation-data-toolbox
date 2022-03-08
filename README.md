# Recommendation project: Data Toolbox

## `recommendation_data_toolbox` package

## Command-line utilities

### Prerequisites

- python3

### Setup in local machine

1. In terminal, navigate to the folder in which the repository will be located. Clone the repository here.

   ```
   git clone https://github.com/nusbeelab/recommendation-data-analysis.git
   ```

1. Change the current directory to the directory of the cloned repository.

   ```
   cd recommendation-data-analysis
   ```

1. Create a virtual environment named `venv` to manage dependencies for the project.

   ```
   python3 -m venv venv
   ```

1. Activate the virtual environment that has been created. Once the virtual environment is active, the prompt in the terminal will begin with `(venv)`.

   On Windows,

   ```
   venv\Scripts\activate.bat
   ```

   On Unix or MacOS,

   ```
   source venv/bin/activate
   ```

1. Install dependencies
   ```
   pip3 install -r requirements.txt
   ```

### Usage

#### Estimation of parameters by command line

```
python3 -m param_estimation generate_intermediate_data --experiment-number <experiment-number>

```

```

python3 -m param_estimation estimate_params --experiment-number <experiment-number> --model <model> [--per-subject] [--include-neg-domain]

```
