#!/bin/bash
python -m venv env
source env/Scripts/activate
pip install -r requirements.txt
pip install -e .
