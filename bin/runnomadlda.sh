#!/bin/bash

trap "echo 'signal.....quit'; exit" SIGHUP SIGINT SIGTERM

. ~/bin/tbbvars.sh intel64
export PATH=$HOME/local/mpich2/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/mpich2/lib/:$LD_LIBRARY_PATH

bindir=`dirname $0`
homedir=`dirname $bindir`
dataroot=$homedir/data/
nomadldabin=$NOMADLDABIN

host_list=""

#
# init
#
init()
{
    #set alpha=50
    #alpha=`echo "scale=4; 50/$topic" | bc`
    #echo "init:set alpha to 50/K = $alpha"
    alpha=50
    datadir=$dataroot/$dataset/nomadlda/

    case $dataset in
        nytimes)
            #alpha=0.1
            beta=0.01
            num_vocabs=111400
            max_doc=300000
            data_cap=800
            ;;
        pubmed2m)
            #alpha=0.1
            beta=0.01
            num_vocabs=144400
            max_doc=2100000
            data_cap=1500
            ;;
        pubmed)
            #alpha=0.1
            beta=0.01
            num_vocabs=144400
            max_doc=8300000
            data_cap=6200
            ;;
        enwiki)
            #alpha=0.1
            beta=0.01
            num_vocabs=144400
            max_doc=8300000
            data_cap=6200
            ;;
 
        *) echo "unkown dataset: $dataset"; help; exit 1;;
    esac
        
    #init the cluster host list
    #hostipfile="cluster-"$nodes".ip"
    hostipfile="cluster.ip"
    if [ ! -f $hostipfile ] ; then
        echo "=================================="
        echo "ERROR: $hostipfile not exist, quit"
        echo "=================================="
        exit -2
    fi
    cathosts="cat $hostipfile"
    hosts=`$cathosts`
    echo $hosts
    id=1
    #for host in `cat conf/$cluster.hostname`; do
    for host in $hosts; do
        if [ $id -eq '1' ]; then
            ((id++))
            host_list=$host
        else
            host_list=$host_list","$host
        fi
        lasthost=$host
    done
    #host_list=$host_list","$lasthost","$lasthost
    #echo "host_list as:"$host_list

}


# runnomadlda 
runnomadlda()
{
    #echo "Start: $cmd"
    logfile="nomadlda_"$dataset"_t"$topic"_"$nodes"x"$threads"_i"$iter"_"$alpha"_"$beta"_t"$timeout"_$1.log"
    
    #cmd="mpiexec -f cluster.ip -n $nodes $bindir/$nomadldabin -T $timeout -k $topic -a $alpha -b $beta -t $iter -n $threads $datadir"
    cmd="mpiexec -n $nodes $bindir/$nomadldabin -T $timeout -k $topic -a $alpha -b $beta -t $iter -n $threads $datadir"
    echo $cmd

    export MV2_ENABLE_AFFINITY=0
    #mpiexec -f cluster.ip -n $nodes $bindir/$nomadldabin -T $timeout -k $topic -a $alpha -b $beta -t $iter -n $threads $datadir 2>&1 | tee $logfile
    mpiexec -n $nodes $bindir/$nomadldabin -T $timeout -k $topic -a $alpha -b $beta -t $iter -n $threads $datadir 2>&1 | tee $logfile
}

#
# main
#
if [ $# -eq 0 ]; then
    echo "runnomadlda.sh <dataset> <iters> <topics> <nodes> <threads> <timeout> <runid>"
else
    dataset=$1
    iter=$2
    topic=$3
    nodes=$4
    threads=$5
    timeout=$6
    runid=$7

    echo $dataset

    if [ -z $thread ]; then
        thread=1
    fi
    if [ -z $iter ] ; then
        iter=100
    fi
    if [ -z $topic ] ; then
        topic=1000
    fi
    if [ -z $nodes ] ; then
        nodes=1
    fi
    if [ -z $timeout ] ; then
        timeout=10
    fi
    if [ -z "$dataset" ] ; then
        dataset=nytimes
    fi
    if [ -z "$runid" ] ; then
        runid=`date +%m%d%H%M%S`
    fi

    init

    # run experiments
    runnomadlda $runid
fi

