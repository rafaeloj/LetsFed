FROM python:3.12-slim

RUN mkdir app
RUN mkdir app/strategies
RUN pip install --upgrade pip
RUN pip install numpy tensorflow tensorflow-cpu flwr scikit-learn
CMD ["python", "-u", "/app/strategies_manager.py"]