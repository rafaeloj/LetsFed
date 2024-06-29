#!/bin/bash

cases=(
    'FedCIA debug set1_var1'
    'FedCIA debug set1_var2'
    'FedCIA debug set1_var3'
    'FedCIA debug set1_var4'
    'FedCIA debug set1_var5'
    'FedCIA debug set1_var6'
    'FedCIA debug set1_var7'
    'FedCIA debug set1_var8'
    'FedCIA debug set1_var9'
    'FedCIA debug set1_var10'
    'FedCIA debug set1_var11'
    'FedCIA debug set1_var12'
    'FedAVG debug'
    'FedDEEV debug'
    'FedPOC debug'
)

file_docker_compose=(
    'dockercompose-cia-mnist-default-c100-r200-le2-p0.20-exp0.30-lsf0.30-dec0.02-thr0.1.yaml'
    'dockercompose-cia-mnist-default_1-c100-r200-le3-p0.25-exp0.25-lsf0.25-dec0.03-thr0.5.yaml'
    'dockercompose-cia-mnist-r_robin-c100-r200-le4-p0.30-exp0.35-lsf0.35-dec0.01-thr1.0.yaml'
    'dockercompose-cia-mnist-default-c100-r200-le5-p0.10-exp0.10-lsf0.15-dec0.04-thr0.1.yaml'
    'dockercompose-cia-mnist-default_1-c100-r200-le1-p0.18-exp0.20-lsf0.20-dec0.02-thr0.5.yaml'
    'dockercompose-cia-mnist-r_robin-c100-r200-le2-p0.30-exp0.25-lsf0.30-dec0.03-thr1.0.yaml'
    'dockercompose-cia-mnist-default-c100-r200-le3-p0.15-exp0.15-lsf0.10-dec0.01-thr0.1.yaml'
    'dockercompose-cia-mnist-default_1-c100-r200-le4-p0.22-exp0.40-lsf0.25-dec0.03-thr0.5.yaml'
    'dockercompose-cia-mnist-r_robin-c100-r200-le1-p0.12-exp0.30-lsf0.40-dec0.05-thr1.0.yaml'
    'dockercompose-cia-mnist-default-c100-r200-le5-p0.20-exp0.15-lsf0.15-dec0.01-thr0.1.yaml'
    'dockercompose-cia-mnist-default_1-c100-r200-le2-p0.28-exp0.45-lsf0.35-dec0.01-thr0.5.yaml'
    'dockercompose-cia-mnist-r_robin-c100-r200-le3-p0.25-exp0.50-lsf0.20-dec0.02-thr1.0.yaml'
    'dockercompose-avg-mnist-random-c50-r50-le4-p0.50-exp0.00-lsf0.00-dec0.01-thr1.yaml'
    'dockercompose-deev-mnist-random-c50-r50-le4-p0.50-exp0.00-lsf0.00-dec0.01-thr1.yaml'
    'dockercompose-poc-mnist-random-c50-r50-le4-p0.50-exp0.00-lsf0.00-dec0.01-thr1.yaml'
)
for i in {0..14};
do
    python3 environment.py -e "${cases[$i]}"
    # for j in {0..1};
    # do
        docker compose -f "${file_docker_compose[$i]}" --profile server up &
        sleep 1
        docker compose -f "${file_docker_compose[$i]}" --profile client up
        # ./clear-dockers.sh
    # done
done
