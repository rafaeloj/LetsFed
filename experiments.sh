#!/bin/bash

cases=(
    "POC 10 ic:10"
    "POC 20 ic:10"
    "POC 30 ic:10"
    "AVG 10 ic:10"
    "AVG 20 ic:10"
    "AVG 30 ic:10"
    "DEEV 0.01 ic:10"
    "DEEV 0.005 ic:10"
    "DEEV 0.001 ic:10"
    "R_ROBIN 10 ic:10"
    "R_ROBIN 20 ic:10"
    "R_ROBIN 30 ic:10"
    "POC 10 ic:20"
    "POC 20 ic:20"
    "POC 30 ic:20"
    "AVG 10 ic:20"
    "AVG 20 ic:20"
    "AVG 30 ic:20"
    "DEEV 0.01 ic:20"
    "DEEV 0.005 ic:20"
    "DEEV 0.001 ic:20"
    "R_ROBIN 10 ic:20"
    "R_ROBIN 20 ic:20"
    "R_ROBIN 30 ic:20"
    "POC 10 ic:30"
    "POC 20 ic:30"
    "POC 30 ic:30"
    "AVG 10 ic:30"
    "AVG 20 ic:30"
    "AVG 30 ic:30"
    "DEEV 0.01 ic:30"
    "DEEV 0.005 ic:30"
    "DEEV 0.001 ic:30"
    "R_ROBIN 10 ic:30"
    "R_ROBIN 20 ic:30"
    "R_ROBIN 30 ic:30"
    "POC 10 ic:40"
    "POC 20 ic:40"
    "POC 30 ic:40"
    "AVG 10 ic:40"
    "AVG 20 ic:40"
    "AVG 30 ic:40"
    "DEEV 0.01 ic:40"
    "DEEV 0.005 ic:40"
    "DEEV 0.001 ic:40"
    "R_ROBIN 10 ic:40"
    "R_ROBIN 20 ic:40"
    "R_ROBIN 30 ic:40"
    "POC 10 ic:50"
    "POC 20 ic:50"
    "POC 30 ic:50"
    "AVG 10 ic:50"
    "AVG 20 ic:50"
    "AVG 30 ic:50"
    "DEEV 0.01 ic:50"
    "DEEV 0.005 ic:50"
    "DEEV 0.001 ic:50"
    "R_ROBIN 10 ic:50"
    "R_ROBIN 20 ic:50"
    "R_ROBIN 30 ic:50"
    "LetsFed POC random ic:10 expl:25"
    "LetsFed POC random ic:20 expl:25"
    "LetsFed POC random ic:30 expl:25"
    "LetsFed POC random ic:40 expl:25"
    "LetsFed POC random ic:50 expl:25"
    "LetsFed POC DEEV ic:10 expl:25"
    "LetsFed POC DEEV ic:20 expl:25"
    "LetsFed POC DEEV ic:30 expl:25"
    "LetsFed POC DEEV ic:40 expl:25"
    "LetsFed POC DEEV ic:50 expl:25"
    "LetsFed POC R_ROBIN ic:10 expl:25"
    "LetsFed POC R_ROBIN ic:20 expl:25"
    "LetsFed POC R_ROBIN ic:30 expl:25"
    "LetsFed POC R_ROBIN ic:40 expl:25"
    "LetsFed POC R_ROBIN ic:50 expl:25"
    "LetsFed POC random ic:10 expl:50"
    "LetsFed POC random ic:20 expl:50"
    "LetsFed POC random ic:30 expl:50"
    "LetsFed POC random ic:40 expl:50"
    "LetsFed POC random ic:50 expl:50"
    "LetsFed POC DEEV ic:10 expl:50"
    "LetsFed POC DEEV ic:20 expl:50"
    "LetsFed POC DEEV ic:30 expl:50"
    "LetsFed POC DEEV ic:40 expl:50"
    "LetsFed POC DEEV ic:50 expl:50"
    "LetsFed POC R_ROBIN ic:10 expl:50"
    "LetsFed POC R_ROBIN ic:20 expl:50"
    "LetsFed POC R_ROBIN ic:30 expl:50"
    "LetsFed POC R_ROBIN ic:40 expl:50"
    "LetsFed POC R_ROBIN ic:50 expl:50"
    "LetsFed POC random ic:10 expl:75"
    "LetsFed POC random ic:20 expl:75"
    "LetsFed POC random ic:30 expl:75"
    "LetsFed POC random ic:40 expl:75"
    "LetsFed POC random ic:50 expl:75"
    "LetsFed POC DEEV ic:10 expl:75"
    "LetsFed POC DEEV ic:20 expl:75"
    "LetsFed POC DEEV ic:30 expl:75"
    "LetsFed POC DEEV ic:40 expl:75"
    "LetsFed POC DEEV ic:50 expl:75"
    "LetsFed POC R_ROBIN ic:10 expl:75"
    "LetsFed POC R_ROBIN ic:20 expl:75"
    "LetsFed POC R_ROBIN ic:30 expl:75"
    "LetsFed POC R_ROBIN ic:40 expl:75"
    "LetsFed POC R_ROBIN ic:50 expl:75"
    "LetsFed RANDOM random ic:10 expl:25"
    "LetsFed RANDOM random ic:20 expl:25"
    "LetsFed RANDOM random ic:30 expl:25"
    "LetsFed RANDOM random ic:40 expl:25"
    "LetsFed RANDOM random ic:50 expl:25"
    "LetsFed RANDOM DEEV ic:10 expl:25"
    "LetsFed RANDOM DEEV ic:20 expl:25"
    "LetsFed RANDOM DEEV ic:30 expl:25"
    "LetsFed RANDOM DEEV ic:40 expl:25"
    "LetsFed RANDOM DEEV ic:50 expl:25"
    "LetsFed RANDOM R_ROBIN ic:10 expl:25"
    "LetsFed RANDOM R_ROBIN ic:20 expl:25"
    "LetsFed RANDOM R_ROBIN ic:30 expl:25"
    "LetsFed RANDOM R_ROBIN ic:40 expl:25"
    "LetsFed RANDOM R_ROBIN ic:50 expl:25"
    "LetsFed RANDOM random ic:10 expl:50"
    "LetsFed RANDOM random ic:20 expl:50"
    "LetsFed RANDOM random ic:30 expl:50"
    "LetsFed RANDOM random ic:40 expl:50"
    "LetsFed RANDOM random ic:50 expl:50"
    "LetsFed RANDOM DEEV ic:10 expl:50"
    "LetsFed RANDOM DEEV ic:20 expl:50"
    "LetsFed RANDOM DEEV ic:30 expl:50"
    "LetsFed RANDOM DEEV ic:40 expl:50"
    "LetsFed RANDOM DEEV ic:50 expl:50"
    "LetsFed RANDOM R_ROBIN ic:10 expl:50"
    "LetsFed RANDOM R_ROBIN ic:20 expl:50"
    "LetsFed RANDOM R_ROBIN ic:30 expl:50"
    "LetsFed RANDOM R_ROBIN ic:40 expl:50"
    "LetsFed RANDOM R_ROBIN ic:50 expl:50"
    "LetsFed RANDOM random ic:10 expl:75"
    "LetsFed RANDOM random ic:20 expl:75"
    "LetsFed RANDOM random ic:30 expl:75"
    "LetsFed RANDOM random ic:40 expl:75"
    "LetsFed RANDOM random ic:50 expl:75"
    "LetsFed RANDOM DEEV ic:10 expl:75"
    "LetsFed RANDOM DEEV ic:20 expl:75"
    "LetsFed RANDOM DEEV ic:30 expl:75"
    "LetsFed RANDOM DEEV ic:40 expl:75"
    "LetsFed RANDOM DEEV ic:50 expl:75"
    "LetsFed RANDOM R_ROBIN ic:10 expl:75"
    "LetsFed RANDOM R_ROBIN ic:20 expl:75"
    "LetsFed RANDOM R_ROBIN ic:30 expl:75"
    "LetsFed RANDOM R_ROBIN ic:40 expl:75"
    "LetsFed RANDOM R_ROBIN ic:50 expl:75"
    "LetsFed DEEV random ic:10 expl:25"
    "LetsFed DEEV random ic:20 expl:25"
    "LetsFed DEEV random ic:30 expl:25"
    "LetsFed DEEV random ic:40 expl:25"
    "LetsFed DEEV random ic:50 expl:25"
    "LetsFed DEEV DEEV ic:10 expl:25"
    "LetsFed DEEV DEEV ic:20 expl:25"
    "LetsFed DEEV DEEV ic:30 expl:25"
    "LetsFed DEEV DEEV ic:40 expl:25"
    "LetsFed DEEV DEEV ic:50 expl:25"
    "LetsFed DEEV R_ROBIN ic:10 expl:25"
    "LetsFed DEEV R_ROBIN ic:20 expl:25"
    "LetsFed DEEV R_ROBIN ic:30 expl:25"
    "LetsFed DEEV R_ROBIN ic:40 expl:25"
    "LetsFed DEEV R_ROBIN ic:50 expl:25"
    "LetsFed DEEV random ic:10 expl:50"
    "LetsFed DEEV random ic:20 expl:50"
    "LetsFed DEEV random ic:30 expl:50"
    "LetsFed DEEV random ic:40 expl:50"
    "LetsFed DEEV random ic:50 expl:50"
    "LetsFed DEEV DEEV ic:10 expl:50"
    "LetsFed DEEV DEEV ic:20 expl:50"
    "LetsFed DEEV DEEV ic:30 expl:50"
    "LetsFed DEEV DEEV ic:40 expl:50"
    "LetsFed DEEV DEEV ic:50 expl:50"
    "LetsFed DEEV R_ROBIN ic:10 expl:50"
    "LetsFed DEEV R_ROBIN ic:20 expl:50"
    "LetsFed DEEV R_ROBIN ic:30 expl:50"
    "LetsFed DEEV R_ROBIN ic:40 expl:50"
    "LetsFed DEEV R_ROBIN ic:50 expl:50"
    "LetsFed DEEV random ic:10 expl:75"
    "LetsFed DEEV random ic:20 expl:75"
    "LetsFed DEEV random ic:30 expl:75"
    "LetsFed DEEV random ic:40 expl:75"
    "LetsFed DEEV random ic:50 expl:75"
    "LetsFed DEEV DEEV ic:10 expl:75"
    "LetsFed DEEV DEEV ic:20 expl:75"
    "LetsFed DEEV DEEV ic:30 expl:75"
    "LetsFed DEEV DEEV ic:40 expl:75"
    "LetsFed DEEV DEEV ic:50 expl:75"
    "LetsFed DEEV R_ROBIN ic:10 expl:75"
    "LetsFed DEEV R_ROBIN ic:20 expl:75"
    "LetsFed DEEV R_ROBIN ic:30 expl:75"
    "LetsFed DEEV R_ROBIN ic:40 expl:75"
    "LetsFed DEEV R_ROBIN ic:50 expl:75"
    "LetsFed DEEV-INVERT random ic:10 expl:25"
    "LetsFed DEEV-INVERT random ic:20 expl:25"
    "LetsFed DEEV-INVERT random ic:30 expl:25"
    "LetsFed DEEV-INVERT random ic:40 expl:25"
    "LetsFed DEEV-INVERT random ic:50 expl:25"
    "LetsFed DEEV-INVERT DEEV ic:10 expl:25"
    "LetsFed DEEV-INVERT DEEV ic:20 expl:25"
    "LetsFed DEEV-INVERT DEEV ic:30 expl:25"
    "LetsFed DEEV-INVERT DEEV ic:40 expl:25"
    "LetsFed DEEV-INVERT DEEV ic:50 expl:25"
    "LetsFed DEEV-INVERT R_ROBIN ic:10 expl:25"
    "LetsFed DEEV-INVERT R_ROBIN ic:20 expl:25"
    "LetsFed DEEV-INVERT R_ROBIN ic:30 expl:25"
    "LetsFed DEEV-INVERT R_ROBIN ic:40 expl:25"
    "LetsFed DEEV-INVERT R_ROBIN ic:50 expl:25"
    "LetsFed DEEV-INVERT random ic:10 expl:50"
    "LetsFed DEEV-INVERT random ic:20 expl:50"
    "LetsFed DEEV-INVERT random ic:30 expl:50"
    "LetsFed DEEV-INVERT random ic:40 expl:50"
    "LetsFed DEEV-INVERT random ic:50 expl:50"
    "LetsFed DEEV-INVERT DEEV ic:10 expl:50"
    "LetsFed DEEV-INVERT DEEV ic:20 expl:50"
    "LetsFed DEEV-INVERT DEEV ic:30 expl:50"
    "LetsFed DEEV-INVERT DEEV ic:40 expl:50"
    "LetsFed DEEV-INVERT DEEV ic:50 expl:50"
    "LetsFed DEEV-INVERT R_ROBIN ic:10 expl:50"
    "LetsFed DEEV-INVERT R_ROBIN ic:20 expl:50"
    "LetsFed DEEV-INVERT R_ROBIN ic:30 expl:50"
    "LetsFed DEEV-INVERT R_ROBIN ic:40 expl:50"
    "LetsFed DEEV-INVERT R_ROBIN ic:50 expl:50"
    "LetsFed DEEV-INVERT random ic:10 expl:75"
    "LetsFed DEEV-INVERT random ic:20 expl:75"
    "LetsFed DEEV-INVERT random ic:30 expl:75"
    "LetsFed DEEV-INVERT random ic:40 expl:75"
    "LetsFed DEEV-INVERT random ic:50 expl:75"
    "LetsFed DEEV-INVERT DEEV ic:10 expl:75"
    "LetsFed DEEV-INVERT DEEV ic:20 expl:75"
    "LetsFed DEEV-INVERT DEEV ic:30 expl:75"
    "LetsFed DEEV-INVERT DEEV ic:40 expl:75"
    "LetsFed DEEV-INVERT DEEV ic:50 expl:75"
    "LetsFed DEEV-INVERT R_ROBIN ic:10 expl:75"
    "LetsFed DEEV-INVERT R_ROBIN ic:20 expl:75"
    "LetsFed DEEV-INVERT R_ROBIN ic:30 expl:75"
    "LetsFed DEEV-INVERT R_ROBIN ic:40 expl:75"
    "LetsFed DEEV-INVERT R_ROBIN ic:50 expl:75"
)


# Create a temporary file to store PIDs
pid_file="/tmp/docker_compose_pids.txt"
> "$pid_file"
# if [ -f "logs/c-data.csv" ]; then
#     rm -i logs/c-data.csv
# fi
# if [ -f "logs/s-data.csv" ]; then
#     rm -i logs/s-data.csv
# fi
rm composes_name.txt
# rm -i composes/composes_name.txt
rm -r dockercompose-*
touch composes_name.txt
for i in "${cases[@]}";
do
    ./clear-dockers.sh
    sleep 1
    python3 environment.py -e "$i" >> composes_name.txt
    echo $(tail -n 1 composes_name.txt) 'created'
    nice -n 0 docker compose -f "$(tail -n 1 composes_name.txt)" --profile server up --build &
    echo $! >> "$pid_file"
    sleep 1
    nice -n 0 docker compose -f "$(tail -n 1 composes_name.txt)" --profile client up --build
    echo $! >> "$pid_file"
done
