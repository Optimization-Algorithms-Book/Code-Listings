This directory contains:
Input files for all precedence graphs included in the data sets of:

Scholl, Armin (1993): Data of Assembly Line Balancing Problems. 
  Schriften zur Quantitativen Betriebswirtschaftslehre 16/93, Th Darmstadt.

Format of in2-files:

line 1: number n of tasks

lines 2-n+1: integer task times

lines n+2,...: direct precedence relations in form "i,j"

last line: end mark "-1,-1" (optional)
