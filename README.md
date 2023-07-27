# Simple deep reinforcement learning demo (DRL) using the lunar lander control problem

This demo was prepared to the elective workshop "Artificial Intelligence and Space Robotics" at the Space Studies Program (SSP) 2023 of the International Space Universiry (ISU).

# Requirements

Python version 3

## Windows

Install Windows Subsystem for Linux (WSL) using the command

    wsl --install -d Ubuntu

For more information, check [this link](https://learn.microsoft.com/en-us/windows/wsl/install).

Make sure you have Python venv installed using

    sudo apt update
    sudo apt install python3-virtualenv

For more information, check [this link](https://stackoverflow.com/questions/71818928/python3-10-source-venv-has-changed).

## Linux

Make sure you have Python venv installed using

    sudo apt update
    sudo apt install python3-virtualenv

## macOS

Make sure you have Python 3 installed checking whether the command

    python3 -V

print Python version number (e.g., Python 3.8.9)

# Installation instructions

Clone the github repository using the command

    git clone https://github.com/rventura/demo-drl-lunar-lander.git

After cloning this repository, use the following commands to install requirements into a virtual environment:

    python3 -m venv env
    source env/bin/activate
    pip3 install -r requirements.txt

# Usage

Note that every time you want to run the code, you need to activate the virtual environment using the command, at the directory the virtual environment was created (see previous step):

    source env/bin/activate

To use the code, make sure the current directory is the repository, e.g.,

    cd demo-drl-lunar-lander

at the directory the github repository was cloned.

To train a DRL model, use the command:

    ./demo_dqn.py train

To test the DRL model, use the command:

    ./demo_dqn.py demo

Both of these commands take a second argument that is the DRL model filename (defaults to dqn.model).

# Author contact

Rodrigo Ventura<br/>
Institute for Systems and Robotics - Lisbon<br/>
Instituto Superior Técnico<br/>
rodrigo.ventura (at) isr.tecnico.ulisboa.pt
