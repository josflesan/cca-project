#!/bin/bash

sudo apt update
sudo apt install -y memcached libmemcached-tools
echo "Waiting for Service to Start"
sleep 30
