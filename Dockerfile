FROM ubuntu:20.04
RUN apt update
# optional 
#RUN apt install software-properties-common -y

RUN apt install python3.8 -y
RUN apt install python3-pip -y
# igraphviz requres separate installation
CMD ["yes n | apt install graphviz"] 

COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./src ./scr
COPY ./*.py .
CMD ["python3", "var-par-v1.py"] 
