#!/bin/bash
cd client || { echo "Failed to enter 'clients' directory"; exit 1; }
docker build -f Dockerfile-cpu -t client-flwr-cpu .
cd ..
cd server || { echo "Failed to enter 'server' directory"; exit 1; }
docker build -f Dockerfile-cpu -t server-flwr-cpu .
cd ..