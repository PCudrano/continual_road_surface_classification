#!/bin/bash
strategy= #cumulative # joint  #naive
trap 'for pid in $pids; do kill $pid; done; exit' INT
for i in {0..5}; do
	#echo ~/storage/cl_road_pavement/outputs/run_${strategy}_perm${i}.txt
	run-docker --container_name cudrano_roadpavement_${strategy}_perm${i} 0 48-56 python  main_experiments.py -w -v --dsorder 0 1 2 --perm $i $strategy &
	pids[${i}]=$!
	sleep 5
	docker logs -f cudrano_roadpavement_${strategy}_perm${i} > ~/storage/cl_road_pavement/outputs/run_${strategy}_perm${i}.log &
	echo "Started exp perm $i"
done

for pid in $pids; do
	wait $pid
done

