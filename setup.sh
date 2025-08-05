#!/bin/bash

sudo apt-get update
sudo apt install python3.12-venv
curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
sudo mv bin/micromamba /usr/local/bin/
rm -r bin/
source ~/.bashrc
micromamba activate sticky_venv