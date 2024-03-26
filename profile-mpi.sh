#!/usr/bin/env bash

GENERATIONS=$1
SIZE=$2
DENSITY=$3
SEED=$4
LAB=$5
MIN=$6
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

for i in 1 2 4 8 16 32 64; do
    if [ $i -lt $MIN ]; then
        continue
    fi

    # Each node has 4 cores, thus 4 processes per node
    nodes=$(echo "($i + 3) / 4" | bc)

    echo "MPI with $i processes on $nodes nodes:" >&2
    TIME=$(run srun -n $i -N $nodes -C $LAB ./mpi/life3d-mpi $GENERATIONS $SIZE $DENSITY $SEED)
    echo >&2
    echo -n " (mpi-$i,$TIME)"
done

echo -n "};"
