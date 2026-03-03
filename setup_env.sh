#!/bin/bash
sudo mkdir -p /mnt/azureuser && sudo chown azureuser:azureuser /mnt/azureuser

if [ ! -d "/mnt/azureuser/.venv" ]; then
    echo "Creating venv on /mnt..."
    python3 -m venv /mnt/azureuser/.venv
    ln -sfn /mnt/azureuser/.venv .venv
    source .venv/bin/activate
    TMPDIR=/mnt/azureuser pip install -r requirements.txt --no-cache-dir
else
    echo "venv already exists, activating..."
    ln -sfn /mnt/azureuser/.venv .venv
    source .venv/bin/activate
fi
