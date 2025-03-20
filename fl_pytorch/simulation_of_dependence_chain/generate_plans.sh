#!/usr/bin/env bash

# Software: https://graphviz.org/download/

dot -T pdf gd.txt > gd.pdf
dot -T pdf permk.txt > permk.pdf

dot -T pdf gd_reschedule.txt > gd_reschedule.pdf
dot -T pdf permk_reschedule.txt > permk_reschedule.pdf

