#!/bin/bash

CurDir=$(pwd)
Arch=knl

## get name from arg of script
Name=$1

## actions
action=collect
# action=collect-with

## collect type supported on knl
# type=advanced-hotspots
# type=hotspots
type=general-exploration
# type=memory-access
# type=hpc-performance

# advanced-hotspots     Advanced Hotspots
# concurrency           Concurrency
# cpugpu-concurrency    CPU/GPU Concurrency
# disk-io               Disk Input and Output
# general-exploration   General Exploration
# gpu-hotspots          GPU Hotspots
# hotspots              Basic Hotspots
# hpc-performance       HPC Performance Characterization
# locksandwaits         Locks and Waits
# memory-access         Memory Access
# sgx-hotspots          SGX Hotspots
# tsx-exploration       TSX Exploration
# tsx-hotspots          TSX Hotspots

## collect with type
# type=runsa

## options
# sec=100 2 itr for daal_sgd 10 nodes hugewiki d1000
# sec=150 1 itr for daal_kmeans 10 nodes hugewiki d1000
sec=150  
# profiling mode, native, mixed, auto
mode=mixed 
# unlimited collection data
dlimit=0
# path to src
path_harp=/N/u/fg474admin/lc37/Harp3-DAAL-2017
path_daal=/N/u/fg474admin/lc37/Lib/DAAL2017/__release_tango_lnx/daal/lib/intel64_lin
path_tbb=/opt/intel/compilers_and_libraries_2017/linux/tbb/lib/intel64_lin_mic
path_misc=/N/u/fg474admin/lc37/Lib/DAAL2017/daal-misc/lib
src_path_harp=/N/u/fg474admin/lc37/Harp3-DAAL-2017
src_path_daal=/N/u/fg474admin/lc37/Lib/DAAL2017/daal
src_path_tbb=/opt/intel/compilers_and_libraries_2017/linux/tbb

## knob option
knob_runsa_cache=MEM_LOAD_UOPS_RETIRED.L1_HIT,MEM_LOAD_UOPS_RETIRED.L2_HIT,MEM_LOAD_UOPS_RETIRED.L3_HIT,MEM_LOAD_UOPS_RETIRED.L1_MISS,MEM_LOAD_UOPS_RETIRED.L2_MISS,MEM_LOAD_UOPS_RETIRED.L3_MISS
knob_runsa_avx=INST_RETIRED.ANY,UOPS_EXECUTED.CORE,UOPS_RETIRED.ALL_PS,MEM_UOPS_RETIRED.ALL_LOADS_PS,MEM_UOPS_RETIRED.ALL_STORES_PS,AVX_INSTS.ALL

## result path
obj=R-$Arch-$Name-$action-$type
resDir=/scratch/logs/VTuneRes/$obj
rm -rf $resDir

# check if the flag is triggered by the running program
# trigger_flag=/scratch/fg474admin/LDA-langshi-Exp/profiling/work/vtune-flag.txt
trigger_flag=./vtune-flag.txt
if [ -f $trigger_flag ]; then
    rm $trigger_flag 
    echo "old trigger flag cleaned"
fi

trigger_vtune=0
max_wait_time=0

# launch the program
../bin/profile_warplda.sh &
exec_pid=$!

sleep 10

## get the process name
pid=$(ps -u fg474admin | grep "warplda.noeval" | awk '{ print $1 }')
echo "Profiling pid: $pid"

wait_file() {

    local file="$1"; shift
    local wait_seconds="${1:-10}"; shift # 10 seconds as default timeout
    max_wait_time=$wait_seconds; 
    until test $((wait_seconds--)) -eq 0 -o -f "$file" ; do sleep 1; done
    if (( $wait_seconds != -1 )) ; then
        trigger_vtune=1
    fi
}

wait_file "$trigger_flag" 200

if  (( $trigger_vtune == 1 )) ; then
    echo "vtune triggered"

# # -k event_config=$knob_runsa_cache \
# # -k sampling-interval=1 \
# ## start collect data
# amplxe-cl -$action $type \
#     -mrte-mode=$mode \
#     -data-limit=$dlimit \
#     -duration=$sec \
#     -r $resDir \
#     -search-dir $path_harp \
#     -search-dir $path_daal \
#     -search-dir $path_tbb \
#     -search-dir $path_misc \
#     -source-search-dir $src_path_harp \
#     -source-search-dir $src_path_daal \
#     -source-search-dir $src_path_tbb \
#     -verbose \
#     -target-pid $pid
#     # -- $DAALROOT/examples/cpp/_results/intel_intel64_parallel_a/mf_sgd_batch.exe 0 3 0 0 
#
# ## start generate reports
# report_type=summary
# group=function
#
# amplxe-cl -report $report_type \
#     -result-dir $resDir \
#     -report-output $CurDir/$obj-$report_type-$group.csv -format csv -csv-delimiter comma
#
# report_type=hotspots
# group=thread,function
# amplxe-cl -report $report_type \
#     -result-dir $resDir \
#     -report-output $CurDir/$obj-$report_type-$group.csv -format csv -csv-delimiter comma

else
    echo "vtune not triggered after $max_wait_time seconds elapsed"
fi

wait $exec_pid
echo -e "Profiling Finished"


