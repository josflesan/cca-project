#!/bin/bash

MEMCACHED_IP=100.96.8.2
INTERNAL_AGENT_A_IP=10.0.16.3
INTERNAL_AGENT_B_IP=10.0.16.3

cd ~
cd memcache-perf-dynamic
./mcperf -s $MEMCACHED_IP --loadonly
./mcperf -s $MEMCACHED_IP -a $INTERNAL_AGENT_A_IP -a $INTERNAL_AGENT_B_IP --noload -T 6 -C 4 -D 4 -Q 1000 -c 4 -t 10 --scan 30000:30500:5
