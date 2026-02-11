Core Concepts
=============

The ``fishy-business`` framework is designed around three main pillars: **Configuration**, **Factories**, and **Unified Orchestration**.

Config-Driven Design
--------------------
Unlike many ML projects where hyperparameters are hardcoded or passed through complex nested dictionaries, this project uses a hierarchical YAML-based configuration. 

*   **Global Settings**: Managed by ``TrainingConfig``.
*   **Model Registries**: Defined in ``models.yaml``, mapping model names to their Python class paths.
*   **Dataset Rules**: Defined in ``datasets.yaml``, specifying how each Excel/CSV file should be filtered and encoded.

The Model Factory
-----------------
The Model Factory (``fishy._core.factory``) is the central dispatcher. It allows you to instantiate any model (Deep, Classic, or Evolutionary) using a simple string identifier. This decouples the training logic from specific model architectures, making it trivial to swap a Transformer for a CNN or an OPLS-DA model.

Unified Orchestration
---------------------
The ``UnifiedTrainer`` acts as a high-level manager. It handles:
1.  **Context Management**: Automatic directory creation and log routing.
2.  **Data Flow**: Linking the ``DataModule`` to the training engine.
3.  **Result Persistence**: Saving metrics, figures, and model checkpoints in a standardized format.

By using this architecture, adding a new model usually requires zero lines of change to the training scripts—you simply register it in the YAML config.
