# Simple deep reinforcement learning demo (DRL) using the lunar lander control problem

This demo was prepared to the elective workshop "Artificial Intelligence and Space Robotics" at the Space Studies Program (SSP) 2023 of the International Space Universiry (ISU).

# Requirements

Python version 3

# Installation instructions

After cloning this repository, use the following commands to install requirements into a virtual environment:

    python3 -m venv env
    source env/bin/activate
    pip3 install -r requirements.txt

# Usage

Note that every time you want to run the code, you need to activate the virtual environment using the command:

    source env/bin/activate

To train a DRL model, use the command:

    ./demo_dqn.py train

To test the DRL model, use the command:

    ./demo_dqn.py demo

Both of these commands take a second argument that is the DRL model filename (defaults to dqn.model).

# Author contact

Rodrigo Ventura
Institute for Systems and Robotics - Lisbon
Instituto Superior Técnico
rodrigo.ventura (at) isr.tecnico.ulisboa.pt
