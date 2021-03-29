#!/bin/bash
OutFileName="${PWD##*/}_all_trials.csv"                # Fix the output name
filename="trace_dump.csv"
i=0                                         # Reset a counter

for d in */ ; do
  echo $d
  if [[ $i -eq 0 ]] ; then 
    head -1 "${d}${filename}" > "$OutFileName"   # Copy header if it is the first file
  fi
  tail -n +2 "${d}${filename}" >> "$OutFileName" # Append from the 2nd line each file
  i=$(( $i + 1 ))                           # Increase the counter
done

