#!/bin/bash

CurDir=$(pwd)

Arch=hsw

## get name from arg of script
Name=lightlda.noeval
thread=32

## action
action=collect
# action=collect-with

## collect type supported on haswell
# type=advanced-hotspots
# type=general-exploration
type=concurrency
# type=locksandwaits

# advanced-hotspots    Advanced Hotspots
# concurrency          Concurrency
# cpugpu-concurrency   CPU/GPU Concurrency
# general-exploration  General Exploration
# hotspots             Basic Hotspots
# locksandwaits        Locks and Waits
# memory-access        Memory Access
# tsx-exploration      TSX Exploration
# tsx-hotspots         TSX Hotspots

## collect with type
# type=runsa

## options
sec=120 # around 10 itrs for warplad 8 threads 

# profiling mode, native, mixed, auto
mode=native 
# unlimited collection data
dlimit=0 
path_multiverso=/scratch/fg474admin/LDA-langshi-Exp/ldaprofiling/src/lightlda/multiverso/lib
path_thirdparty=/scratch/fg474admin/LDA-langshi-Exp/ldaprofiling/src/lightlda/multiverso/third_party/lib
path_tbb=/opt/intel/compilers_and_libraries_2016/linux/tbb/lib/intel64_lin/gcc4.4
src_path_multiverso=/scratch/fg474admin/LDA-langshi-Exp/ldaprofiling/src/lightlda/multiverso/src
src_path_thirdparty=/scratch/fg474admin/LDA-langshi-Exp/ldaprofiling/src/lightlda/multiverso/third_party/include
src_path_tbb=/opt/intel/compilers_and_libraries_2016/linux/tbb

## knob option
# knob_runsa_cache=MEM_LOAD_UOPS_RETIRED.L1_HIT,MEM_LOAD_UOPS_RETIRED.L2_HIT,MEM_LOAD_UOPS_RETIRED.L3_HIT,MEM_LOAD_UOPS_RETIRED.L1_MISS,MEM_LOAD_UOPS_RETIRED.L2_MISS,MEM_LOAD_UOPS_RETIRED.L3_MISS
knob_runsa_avx=INST_RETIRED.ANY,UOPS_EXECUTED.CORE,UOPS_RETIRED.ALL_PS,MEM_UOPS_RETIRED.ALL_LOADS_PS,MEM_UOPS_RETIRED.ALL_STORES_PS,AVX_INSTS.ALL

## result path
obj=R-$Arch-$Name-thd$thread-$action-$type-newEnwiki
resDir=/scratch/logs/VTuneRes/$obj
if [ -d $resDir ]; then
    echo "remove existed res folder"
    rm -rf $resDir
fi

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
../bin/profile_lightlda.sh $thread 10 &
exec_pid=$!

# sleep 10
#
# ## get the process name
# pid=$(ps -u fg474admin | grep "$Name" | awk '{ print $1 }')
# echo "Profiling pid: $pid"

wait_file() {

    local file="$1"; shift
    local wait_seconds="${1:-10}"; shift # 10 seconds as default timeout
    max_wait_time=$wait_seconds; 
    until test $((wait_seconds--)) -eq 0 -o -f "$file" ; do sleep 1; done
    echo "current wait_seconds: $wait_seconds"
    if (( $wait_seconds != -1 )); then
        trigger_vtune=1
    fi
}

wait_file "$trigger_flag" 5000

if (( $trigger_vtune == 1 )); then
    echo "vtune triggered"
## get the process name
pid=$(ps -u fg474admin | grep "$Name" | awk '{ print $1 }')
echo "Profiling pid: $pid"

    # -k event-config=$knob_runsa_cache \
    # -k event-config=$knob_runsa_avx \
# start collect data
amplxe-cl -$action $type \
    -mrte-mode=$mode \
    -data-limit=$dlimit \
    -duration=$sec \
    -r $resDir \
    -search-dir $path_multiverso \
    -search-dir $path_thirdparty \
    -search-dir $path_tbb \
    -source-search-dir $src_path_multiverso \
    -source-search-dir $src_path_thirdparty \
    -source-search-dir $src_path_tbb \
    -target-pid $pid

## start generate reports
report_type=summary
group=function

amplxe-cl -report $report_type \
    -result-dir $resDir \
    -report-output $CurDir/$obj-$report_type-$group.csv -format csv -csv-delimiter comma

report_type=hotspots
group=thread,function
amplxe-cl -report $report_type \
    -result-dir $resDir \
    -report-output $CurDir/$obj-$report_type-$group.csv -format csv -csv-delimiter comma
else
    echo "vtune not triggered after $max_wait_time seconds elapsed"
fi

wait $exec_pid
echo -e "Profiling Finished"

#
#
