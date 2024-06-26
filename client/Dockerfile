# # FROM python:3.12-slim
FROM tensorflow/tensorflow:latest-gpu

RUN mkdir client
RUN mkdir client/strategies/
RUN mkdir client/strategies/engagement_strategy
RUN pip install --upgrade pip
RUN pip install flwr-datasets
RUN pip install flwr_datasets[vision]
# RUN python3.12 -m pip install flwr_datasets[audio]
RUN python -m pip install numpy pandas flwr keras scikit-learn matplotlib datasets seaborn

CMD ["python3", "-u", "/client/strategies_manager.py"]
