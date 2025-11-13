#!/bin/bash
docker build -f client/Dockerfile -t client-flwr-cpu .
docker build -f server/Dockerfile -t server-flwr-cpu .
