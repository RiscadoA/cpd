#!/usr/bin/env bash

GENERATIONS=$1
SIZE=$2
DENSITY=$3
SEED=$4
MAX_THREADS=$5
REPEAT_TIMES=3

function run {
    local cmd=$@
    local sum=0
    for i in $(seq 1 $REPEAT_TIMES); do
        local time=$($cmd 2>&1 >/dev/null | cut -d 's' -f 1)
        sum=$(echo $sum + $time | bc)
        echo "Run $i: $time" >&2
    done
    local average=$(echo "$sum $REPEAT_TIMES" | awk '{printf "%.1f", $1 / $2}')
    echo "Average time: $average" >&2
    echo $average
}

echo -n "\addplot coordinates {"

echo "Serial:" >&2
TIME=$(run ./serial/life3d $GENERATIONS $SIZE $DENSITY $SEED)
echo >&2
echo -n "(serial,$TIME)"

for i in 1 2 4 8; do
    echo "OpenMP with $i threads:" >&2
    TIME=$(OMP_NUM_THREADS=$i run ./omp/life3d-omp $GENERATIONS $SIZE $DENSITY $SEED)
    echo >&2
    echo -n " (omp-$i,$TIME)"
done

echo -n "};"
