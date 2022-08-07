# syntax=docker/dockerfile:1
FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install python3.9 -y
RUN apt-get install python3-pip -y

# igraphviz requres a separate installation - takes long
#RUN apt-get install graphviz | yes n

WORKDIR /symindy

COPY requirements.txt .
COPY ./src ./src
COPY ./*.py .

# downgrading of setuptools is required by DEAP
RUN pip3 install setuptools==58.0.0
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

CMD ["python3", "src/validation/reconstruct_cubic_damped_sho.py", "src/validation/reconstruct_cubic_damped_sho.py", \
    "src/validation/reconstruct_lorenz.py", "src/validation/reconstruct_myspring.py", "src/validation/SINDy_vs_SymINDy.py"]
