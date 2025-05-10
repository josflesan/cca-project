#!/bin/bash

CORE_START=0
CORE_END=0

sudo systemctl restart memcached
echo "Waiting for service to restart..."
sleep 60

# Determine the PID of the memcached process
pid=$(pidof memcached)
sudo taskset -a -cp $CORE_START-$CORE_END $pid
