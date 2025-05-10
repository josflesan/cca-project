#!/bin/bash

SERVER=memcache-server-snph

gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing scripts/scheduler.py ubuntu@$SERVER:~/ --zone europe-west1-b
