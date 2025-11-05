# have to edit location of service file in below 2 lines if moved
sudo rm /etc/systemd/system/arc.service
sudo ln -s /var/www/arc/src/api/arc.service /etc/systemd/system/arc.service
sudo systemctl daemon-reload
sudo systemctl restart arc.service