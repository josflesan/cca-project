#!/bin/bash

MEMCACHED_IP=10.0.16.6
INTERNAL_AGENT_IP=10.0.16.5

cd ~
cd memcache-perf-dynamic
./mcperf -s $MEMCACHED_IP --loadonly
./mcperf -s $MEMCACHED_IP -a $INTERNAL_AGENT_IP --noload -T 8 -C 8 -D 4 -Q 1000 -c 8 -t 10 --qps_interval 2 --qps_min 5000 --qps_max 180000
