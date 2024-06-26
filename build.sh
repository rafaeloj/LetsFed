#!/bin/bash
cd clients || { echo "Failed to enter 'clients' directory"; exit 1; }
docker build -t client-flwr .
cd ..
cd server || { echo "Failed to enter 'server' directory"; exit 1; }
docker build -t server-flwr .
cd ..