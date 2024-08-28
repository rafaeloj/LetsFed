import os
import yaml
from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess
import glob


app = Flask(__name__)
app.secret_key = "supersecretkey"  # Chave para mensagens flash
ROOT = "/home/ozymandias/msc/repositories/fedcia"
generated_compose_file = None
def get_yaml_files(folder):
    yaml_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.yaml'):
                yaml_files.append(os.path.join(root, file))
    return yaml_files

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            flash(f"Erro ao carregar o arquivo YAML: {e}")
            return {}

def save_yaml(file_path, data):
    with open(file_path, 'w') as file:
        try:
            yaml.safe_dump(data, file, default_flow_style=False, allow_unicode=True, explicit_end=False)
        except yaml.YAMLError as e:
            flash(f"Erro ao salvar o arquivo YAML: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    yaml_files = get_yaml_files(f"{ROOT}/conf")
    yaml_contents = {file: load_yaml(file) for file in yaml_files}

    if request.method == 'POST':
        for file_path in yaml_files:
            new_data = request.form.get(f'content_{file_path}', '')
            try:
                data = yaml.safe_load(new_data)
                save_yaml(file_path, data)
            except yaml.YAMLError as e:
                flash(f"Erro ao processar o YAML do arquivo {file_path}: {e}")
        flash("Alterações salvas com sucesso!")
        return redirect(url_for('index'))

    # Passando a função yaml.dump para o template
    return render_template('edit_all.html', yaml_contents=yaml_contents, yaml_dump=yaml.dump)

@app.route('/execute_environment', methods=['POST'])
def execute_environment():
    try:
        # Executa o comando "python3 environment.py" no diretório atual
        result = subprocess.run(['python3', 'environment.py'], capture_output=True, text=True, cwd=ROOT)
        # Captura o arquivo docker-compose gerado
        generated_compose_file = get_latest_compose_file()

        if result.returncode == 0:
            flash(f"Comando executado com sucesso: {result.stdout}")
        else:
            flash(f"Erro ao executar o comando: {result.stderr}")

    except Exception as e:
        flash(f"Erro ao executar o comando: {str(e)}")

    return redirect(url_for('index'))  # Redireciona para a página principal


@app.template_filter('basename')
def basename_filter(path):
    return os.path.basename(path)

# Função para capturar o arquivo mais recente gerado com extensão .yml ou .yaml
def get_latest_compose_file():
    files = glob.glob('*.yml') + glob.glob('*.yaml')
    if files:
        # Ordena os arquivos pelo tempo de modificação, pegando o mais recente
        latest_file = max(files, key=os.path.getmtime)
        return latest_file
    return None

# Rota para executar o arquivo docker-compose gerado
@app.route('/execute_docker_compose', methods=['POST'])
def execute_docker_compose():
    global generated_compose_file
    print("ENTROU")
    if generated_compose_file and os.path.exists(generated_compose_file):
        try:
            print(generated_compose_file)
            result = subprocess.run(['docker', 'compose', '-f', generated_compose_file, '--profile','server', 'up', '-d'])
            result = subprocess.run(['docker', 'compose', '-f', generated_compose_file, '--profile','client', 'up', '-d'])

            if result.returncode == 0:
                flash("Docker Compose executado com sucesso!")
            else:
                flash(f"Erro ao executar Docker Compose: {result.stderr}")

        except Exception as e:
            flash(f"Erro ao executar Docker Compose: {str(e)}")
    else:
        flash("Nenhum arquivo docker-compose válido encontrado.")

    return redirect(url_for('index'))
if __name__ == '__main__':
    app.run(debug=True)
