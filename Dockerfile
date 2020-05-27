FROM python:3.6.10-stretch
RUN pip install --upgrade pip
RUN pip install setuptools numpy scipy scikit-learn==0.23 cython matplotlib pandas numpy guppy3
RUN pip install git+https://github.com/scikit-garden/scikit-garden.git
WORKDIR /home