# 🚀 Transformer Fault Diagnosis

This repository contains a full pipeline for simulating motor faults, generating labeled time-series data, and training a Transformer-based fault diagnosis model.


## 📁 Project Structure

```plaintext
transformer_fault_diagnosis/
│
├── data/                                     # Data storeage
│   ├── desired_SE(3), q_d, lambda_d          # Store desired SE(3) matrix, desired joint angle, desired motor thrust (before fault)
│   ├── actual_SE(3), q_a, lambda_a           # Store actual trajectory, actual joint angle, actual motor thrust (after fault)
│   └── label                                 # Store label is type of fault for each link, motor / label is described as matrix
│
├── simulation/                               # Data generation
│   ├── __init__.py
│   ├── generate_dataset.py                   # Entire pipeline
│   ├── generate_trajectory.py                # Generate desired SE(3) matrix
│   ├── inverse_kinematics.py                 # Generate desired joint angle from desired SE(3) matrix
│   ├── inverse_motor_dynamics.py             # Generate desired motor thrust from desired joint angle
│   ├── fault_injection.py                    # Inject faults into motor thrust by applying a scaled product and adding noise
│   ├── forward_motor_dynamics.py             # Generate actual joint angle from fault(actual) motor thrust
│   └── forward_kinematics.py                 # Generate actual SE(3) matrix
│
├── model/                                    # Transformer model and training/evaluation modules
│   ├── __init__.py                
│   ├── Transformer.py                        # Positional encoding, Transformer encoder, and classification head
│   ├── train.py                              # Training loop for the Transformer model
│   └── evaluate.py                           # Evaluation metrics: accuracy, F1 score, confusion matrix, etc.
│
├── utils/                                    # Utility functions
│   ├── __init__.py               
│   ├── io.py                                 # File I/O functions (e.g., load/save CSV or NumPy data)
│   ├── config.py                             # Configuration and hyperparameter settings
│   └── visualization.py                      # Visualization utilities for trajectories and results
│
├── main.py                                   # Main script to run the entire pipeline (from data generation to model evaluation)
├── requirements.txt         
└── README.md                     