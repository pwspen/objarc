# have to edit location of service file in below 2 lines if moved
sudo rm /etc/systemd/system/arc-api.service
sudo ln -s /home/synapso/objarc/arc_api.service /etc/systemd/system/arc-api.service
sudo systemctl daemon-reload
sudo systemctl restart arc-api.service