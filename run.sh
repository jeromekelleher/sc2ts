#!/bin/bash
set -e
set -u
# set -x

mismatches=3
max_daily_samples=1000
num_threads=8

# Paths
# datadir=testrun
datadir=data/viridian
run_id=tmp-dev-hp
# run_id=upgma-mds-$max_daily_samples-md-$max_submission_delay-mm-$mismatches
resultsdir=results/$run_id
results_prefix=$resultsdir/$run_id-
logfile=logs/$run_id.log

alignments=$datadir/alignments.db
metadata=$datadir/metadata.db
matches=$resultsdir/matches.db

options="--num-threads $num_threads -vv -l $logfile "
# options+="--max-submission-delay $max_submission_delay "
# options+="--max-daily-samples $max_daily_samples "
options+="--num-mismatches $mismatches"

mkdir -p $resultsdir

# date=2000-01-01
# last_ts=$resultsdir/initial.ts
# python3 -m sc2ts initialise $last_ts $matches

date=2020-03-01
last_ts="$results_prefix$date".ts

dates=`python3 -m sc2ts list-dates --after $date $metadata | grep -v 2021-12-31`
for date in $dates; do
    out_ts="$results_prefix$date".ts
    python3 -m sc2ts extend $last_ts $date $alignments $metadata \
        $matches $out_ts $options
    last_ts=$out_ts
    # break
done
