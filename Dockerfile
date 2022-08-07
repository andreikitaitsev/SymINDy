# TODO UPDATE
FROM ubuntu:20.04
RUN apt update
WORKDIR symindy
# optional
#RUN apt install software-properties-common -y

RUN apt install python3.9 -y
RUN apt install python3-pip -y
# igraphviz requres a separate installation
RUN apt install graphviz | yes n

COPY requirements.txt .
COPY ./src ./src
COPY ./*.py .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -e .

CMD ["python3", "src/validation/reconstruct_cubic_damped_sho.py", "src/validation/reconstruct_cubic_damped_sho.py", \
    "src/validation/reconstruct_lorenz.py", "src/validation/reconstruct_myspring.py", "src/validation/SINDy_vs_SymINDy.py"]
