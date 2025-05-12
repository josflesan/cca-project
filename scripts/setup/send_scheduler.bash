#!/bin/bash

SERVER=memcache-server-z7lt

gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing scripts/scheduler_v2.py ubuntu@$SERVER:~/src/scheduler.py --zone europe-west1-b
gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing scripts/strategies.py ubuntu@$SERVER:~/src --zone europe-west1-b
gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing scripts/utils/scheduler_logger.py ubuntu@$SERVER:~/src/utils --zone europe-west1-b
gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing scripts/utils/scheduler_utils.py ubuntu@$SERVER:~/src/utils --zone europe-west1-b
