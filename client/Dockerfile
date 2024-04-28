FROM python:3.12-slim

RUN mkdir client
RUN mkdir client/strategies/
RUN mkdir client/strategies/engagement_strategy
RUN pip install --upgrade pip
RUN pip install flwr-datasets
RUN pip install flwr_datasets[vision]
RUN pip install flwr_datasets[audio]
RUN pip install numpy pandas tensorflow tensorflow-cpu flwr scikit-learn matplotlib datasets seaborn

CMD ["python", "-u", "/client/strategies_manager.py"]