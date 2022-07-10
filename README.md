# Foobar

NExG is a platform for performing state space exploration in closed loop control systems.

## Installation

```
sudo apt install python3
sudo apt install python3-pip
```

Use the link [notebook-ubuntu22](https://linuxhint.com/install-jupyter-notebook-ubuntu-22-04/) to install jupyter notebook.
Create a virtual environment and install the following packages.

```
pip3 install matplotlib
pip3 install tensorflow
pip3 install scipy
pip3 install pyyaml
pip3 install shapely
pip3 install scikit-learn
pip3 install --upgrade traitlets
pip3 install cuda-python
```

Once done with the installation, open a terminal and set the environment variable "NXG_PATH"
to the NExG directory.

```
export NXG_PATH=”/home/.../NExG”
```

Finally, run the following command and copy the localhost link to open in the browser.

```
jupyter notebook
```