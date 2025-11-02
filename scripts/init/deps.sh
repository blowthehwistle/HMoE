conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV_NAME="hmoe"
PYTHON_VERSION="3.12"

if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Environment ${ENV_NAME} already exists, skipping creation step"
else
    echo "Creating environment ${ENV_NAME}..."
    conda create --name $ENV_NAME python=$PYTHON_VERSION -y
fi

conda activate $ENV_NAME

pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall

pip install --no-cache-dir transformers torchdata datasets tiktoken einops