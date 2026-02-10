To run the experiments, please follow these steps:

1.  **Install Dependencies:**
    It appears that the previous attempt to install dependencies was successful, but if you encounter any issues, you can manually install them by running the following command in your terminal from the project root directory:

    ```bash
    python3 -m pip install .
    ```

    This command will install all necessary packages listed in `pyproject.toml` in editable mode. You might also want to upgrade pip to the latest version by running:

    ```bash
    /Library/Developer/CommandLineTools/usr/bin/python3 -m pip install --upgrade pip
    ```

2.  **Run the Experiment Script:**
    Once the dependencies are installed, you can run the experiment script I've created by executing the following command from your project root directory:

    ```bash
    python3 run_all_experiments.py
    ```

The script `run_all_experiments.py` will iterate through all specified datasets and models, running both classic and deep learning experiments. Please note that the `epochs` for deep learning models have been set to 10 for quicker demonstration; you might want to adjust this value within `run_all_experiments.py` for more thorough training.