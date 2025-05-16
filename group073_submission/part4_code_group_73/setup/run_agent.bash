#!/bin/bash

TARG=8

cd ~
cd memcache-perf-dynamic
./mcperf -T $TARG -A
