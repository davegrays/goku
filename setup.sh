env_name=goku
python_version=3.10

conda create -n ${env_name} python=${python_version}
source activate ${env_name}
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --name ${env_name}
source deactivatex
