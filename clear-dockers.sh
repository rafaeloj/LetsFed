#!/bin/bash

# Listar todos os contêineres em execução
containers=$(docker ps -a --format "{{.ID}} {{.Image}} {{.Names}}")

# Para cada contêiner na lista
while IFS= read -r container; do
    # Extrair o ID, a imagem e o nome do contêiner
    container_id=$(echo "$container" | awk '{print $1}')
    container_image=$(echo "$container" | awk '{print $2}')
    container_name=$(echo "$container" | awk '{print $3}')

    # Verificar se o nome do contêiner contém uma palavra-chave relacionada ao federated learning
    if [[ "$container_name" == *"fedcia"* || "$container_name" == *"rfl_server"* || "$container_name" == *"rfl_client"* ]]; then
        # Remover o contêiner
        docker rm -f "$container_id"
    fi
done <<< "$containers"
