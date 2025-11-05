#!/bin/bash
set -e
# Repo must be installed in /var/www/arc/ !!!

# Build frontend
cd /var/www/arc/viz
npm install
npm run build

#!/bin/bash
set -euo pipefail

# Ensure repo path and ownership for the daemon user
install -d -o www-data -g www-data /var/www/arc
chown -R www-data:www-data /var/www/arc

# Install uv system-wide if missing
if ! command -v uv >/dev/null 2>&1; then
  # Install into /usr/local/bin so all users can access it
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
fi

# Create venv and install Python deps as www-data
sudo -u www-data -H bash -lc '
  cd /var/www/arc
  uv venv .venv
  uv sync
'

# Install and manage systemd service
cd /var/www/arc/src/api/
cp arc.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable arc
systemctl restart arc

# Don't fail the script if status returns non-zero
systemctl status arc || true