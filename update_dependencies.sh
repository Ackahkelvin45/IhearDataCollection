#!/bin/bash

echo "Updating dependencies to fix Numba/NumPy compatibility issues..."

# Deactivate virtual environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating virtual environment..."
    deactivate
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Uninstall problematic packages
echo "Uninstalling problematic packages..."
pip uninstall -y numba numpy

# Install compatible versions
echo "Installing compatible versions..."
pip install numba==0.58.1 numpy==1.24.3

# Verify installation
echo "Verifying installation..."
python -c "import numba; import numpy; print(f'Numba version: {numba.__version__}'); print(f'NumPy version: {numpy.__version__}')"

echo "Dependencies updated successfully!"
echo "You can now restart your Celery worker to test the audio processing." 