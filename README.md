# 🚀 Transformer Fault Diagnosis

This repository contains a full pipeline for simulating motor faults, generating labeled time-series data, and training a Transformer-based fault diagnosis model.


## 📁 Project Structure

```plaintext
transformer_fault_diagnosis/
│
├── data/                                       # Data storage
│   ├── desired_SE(3), q_d, lambda_d            # Desired SE(3), joint angle, motor thrust (before fault)
│   ├── actual_SE(3), q_a, lambda_a             # Actual SE(3), joint angle, motor thrust (after fault)
│   └── label                                   # Fault labels per link/motor
│
├── simulation/                                 # Data generation pipeline
│   ├── __init__.py
│   ├── generate_dataset.py                     # Full data generation pipeline
│   ├── generate_trajectory.py                  # Generate desired SE(3) matrix
│   ├── inverse_kinematics.py                   # Convert SE(3) to joint angles
│   ├── inverse_motor_dynamics.py               # Compute motor thrust from joint angles
│   ├── fault_injection.py                      # Inject scaled & noisy faults into motor thrust
│   ├── forward_motor_dynamics.py               # Simulate actual joint angles after fault
│   └── forward_kinematics.py                   # Generate actual SE(3) matrix
│
├── model/                                      # Transformer model & training/evaluation
│   ├── __init__.py
│   ├── Transformer.py                          # Positional encoding, encoder, classifier
│   ├── train.py                                # Training loop
│   └── evaluate.py                             # Accuracy, F1 score, confusion matrix
│
├── utils/                                      # Utility modules
│   ├── __init__.py
│   ├── io.py                                   # File I/O for CSV, NumPy, etc.
│   ├── config.py                               # Hyperparameter and config settings
│   └── visualization.py                        # Plotting and result visualization
│
├── main.py                                     # Main script to run the full pipeline
├── requirements.txt                            # Required packages
└── README.md
