#!/bin/bash
set -e

# Repo must be installed in /var/www/arc/ !!!

cd /var/www/arc/viz
npm run build

cd /var/www/arc
uv venv .venv
uv sync
sudo cp arc.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable arc
sudo systemctl restart arc
sudo systemctl status arc