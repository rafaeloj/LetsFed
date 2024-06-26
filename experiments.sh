#!/bin/bash

cases=(
    'FedAVG debug'
    'FedCIA debug'
    'FedDEEV debug'
    'FedPOC debug'
)

file_docker_compose=(
    'dockercompose-avg-cifar10-random-c25-r200-e0.40-d0.1.yaml'
    'dockercompose-cia-cifar10-default_1-c25-r200-e0.40-d0.1.yaml'
    'dockercompose-deev-cifar10-random-c25-r200-e0.40-d0.1.yaml'
    'dockercompose-poc-cifar10-random-c25-r200-e0.40-d0.1.yaml'
)
for i in {0..4};
do
    python3 environment.py -e "${cases[$i]}"
    for j in {0..10};
    do
        docker compose -f "${file_docker_compose[$i]}" --profile server up &
        sleep 1
        docker compose -f "${file_docker_compose[$i]}" --profile client up
    done
done
