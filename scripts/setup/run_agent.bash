#!/bin/bash

TARG=4

cd ~
cd memcache-perf-dynamic
./mcperf -T $TARG -A
