#!/bin/bash

# Definindo o array de casos
cases=(
    'final | avg | mnist | none | 50 | 100 | 0.25 | 0.1 | Seleção random 0.25'
    'final | avg | mnist | none | 50 | 100 | 0.50 | 0.1 | Seleção random 0.25'
    'final | avg | mnist | none | 50 | 100 | 0.75 | 0.1 | Seleção random 0.25'
    'final | poc | mnist | none | 50 | 100 | 0.25 | 0.1 | Perc 0.25'
    'final | poc | mnist | none | 50 | 100 | 0.50 | 0.1 | Perc 0.25'
    'final | poc | mnist | none | 50 | 100 | 0.75 | 0.1 | Perc 0.25'
    'final | deev | mnist | none | 50 | 100 | 0.25 | 0.1 | decay 0.005'
    'final | deev | mnist | none | 50 | 100 | 0.50 | 0.1 | decay 0.005'
    'final | deev | mnist | none | 50 | 100 | 0.75 | 0.1 | decay 0.005'
)
for i in {1..5};
do
    for case in "${cases[@]}"
    do
        # Extrair partes da string de caso
        strategy=$(echo "$case" | cut -d '|' -f 2 | xargs)
        dataset=$(echo "$case" | cut -d '|' -f 3 | xargs)
        selection=$(echo "$case" | cut -d '|' -f 4 | xargs)
        n_clients=$(echo "$case" | cut -d '|' -f 5 | xargs)
        rounds=$(echo "$case" | cut -d '|' -f 6 | xargs)
        engaged_clients=$(echo "$case" | cut -d '|' -f 7 | xargs)
        dirichlet=$(echo "$case" | cut -d '|' -f 8 | xargs)

        # # Calcular o valor de engajamento em decimal

        # # Formatar o nome do arquivo YAML
        file_name="dockercompose-${strategy}-${dataset}-${selection}-c${n_clients}-r${rounds}-e${engaged_clients}-d${dirichlet}.yaml"
        echo "Generating file: ${file_name}"
        # echo $case
        # # Executar o comando Python para gerar o arquivo
        python3.8 environment.py -e "$case"
        # echo "$file_name"
        # Limpar dockers e executar docker compose
        ./clear-dockers.sh && docker compose -f "$file_name" --profile server up -d
        sleep 2
        start=`date +%s%N`
            docker compose -f "$file_name" --profile client up
        end=`date +%s%N`
        final_time = `expr $end - $start`
        echo "$i,$case,$final_time\n" >> logs/times.csv
        # # # Aguardar 30 segundos antes de prosseguir para o próximo caso
    done
done