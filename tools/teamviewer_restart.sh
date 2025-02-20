#!/bin/bash

# Wait for network to be available
while ! ping -c 1 -W 1 8.8.8.8; do
    sleep 1
done

# Start TeamViewer
teamviewer &

# Wait for TeamViewer to start and generate a new password
sleep 30

# Get the new TeamViewer password
PASSWORD=$(teamviewer info | grep "Password:" | awk '{print $2}')

# Send email with the new password
echo "New TeamViewer password: $PASSWORD" | mail -s "Jetson Nano TeamViewer Password" netcat@gmail.com



