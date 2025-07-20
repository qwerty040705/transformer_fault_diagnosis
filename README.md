# 🚀 Transformer Fault Diagnosis

This repository contains a full pipeline for simulating motor faults, generating labeled time-series data, and training a Transformer-based fault diagnosis model.


## 📁 Project Structure

```plaintext
transformer_fault_diagnosis/
│
├── data/                                     # Data storage
│   ├── desired_SE(3), q_d, lambda_d          # Desired SE(3) matrix, joint angles, and motor thrust (before fault)
│   ├── actual_SE(3), q_a, lambda_a           # Actual trajectory, joint angles, and motor thrust (after fault)
│   └── label                                 # Fault labels for each link and motor, stored as matrices
│
├── simulation/                               # Data generation modules
│   ├── __init__.py
│   ├── generate_dataset.py                   # Full pipeline for data generation
│   ├── generate_trajectory.py                # Generate desired SE(3) trajectory
│   ├── inverse_kinematics.py                 # Compute joint angles from desired SE(3) trajectory
│   ├── inverse_motor_dynamics.py             # Compute motor thrust from joint angles (pre-fault)
│   ├── fault_injection.py                    # Apply fault to motor thrust (scaling, noise) and generate labels
│   ├── forward_motor_dynamics.py             # Compute joint angles from faulted motor thrust
│   ├── forward_kinematics.py                 # Compute actual SE(3) trajectory from joint angles
│   └── control/                              # Control logic for generating wrench or joint commands
│       ├── __init__.py
│       ├── impedance_controller.py           # End-effector impedance control (force/moment output)
│       ├── centralized_controller.py         # Multi-joint centralized controller
│       ├── selective_mapping.py              # Directional filtering for force/moment commands
│       └── clik_controller.py                # Closed-loop inverse kinematics solver
│
├── model/                                    # Transformer model and training/evaluation modules
│   ├── __init__.py
│   ├── Transformer.py                        # Positional encoding, Transformer encoder, classification head
│   ├── train.py                              # Training loop for Transformer model
│   └── evaluate.py                           # Evaluation metrics: accuracy, F1 score, confusion matrix, etc.
│
├── utils/                                    # Utility functions
│   ├── __init__.py
│   ├── io.py                                 # File I/O operations (e.g., CSV or NumPy)
│   ├── config.py                             # Configuration and hyperparameter settings
│   └── visualization.py                      # Plotting utilities for trajectories and results
│
├── main.py                                   # Main script to run the full pipeline (data → training → evaluation)
├── requirements.txt                          # Python package dependencies
└── README.md                                 # Project documentation
