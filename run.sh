#!/bin/bash
trap 'for pid in $pids; do kill $pid; done; exit' INT
run () {
	strategy=$1
	i=$2
	#echo ~/storage/cl_road_pavement/outputs/run_${strategy}_perm${i}.txt
	run-docker --container_name cudrano_roadpavement_${strategy}_perm${i} 7 104-111 python main_experiments.py -w -v --dsorder 0 1 2 --perm $i $strategy &
	pids[${i}]=$!
	sleep 5
	docker logs -f cudrano_roadpavement_${strategy}_perm${i} > ~/storage/cl_road_pavement/outputs/run_${strategy}_perm${i}.log &
	echo "Started exp perm $i"
}

run naive 0

for pid in $pids; do
	wait $pid
done

