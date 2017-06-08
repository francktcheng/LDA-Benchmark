#!/bin/bash
bindir=`dirname $0`
homedir=`dirname $bindir`
dataroot=$homedir/data/
trainer=warplda.noeval

#
# init
#
init()
{
    #set alpha=50/K
    alpha=50
#alpha=`echo "scale=4; 50/$topic" | bc`
#    echo "init:set alpha to 50/K = $alpha"
    datadir=$dataroot/$dataset/warplda-$nodes/train
    beta=0.01
}


# 4. Run LightLDA
runwarplda()
{
    cmd="$bindir/$trainer --niter $iter --mh $mh_step --prefix $datadir"
    echo "Start: $cmd"

    logfile="warplda_"$dataset"_t"$topic"_"$nodes"x"$threads"_i"$iter"_"$alpha"_"$beta"_"$mh_step"_$1.log"

    export OMP_NUM_THREADS=$threads
    #$bindir/warplda --niter $iter --mh $mh_step --prefix $datadir --dumpmodel false --beta $beta --alpha $alpha --dumpz false -k $topic | tee $logfile
    $bindir/$trainer --niter $iter --mh $mh_step --prefix $datadir --beta $beta --alpha $alpha --k $topic | tee $logfile

}

#
# main
#
if [ $# -eq 0 ]; then
    echo "runwarplda.sh <dataset> <iters> <topics> <nodes> <threads> <mh_steps> <runid>"
else
    dataset=$1
    iter=$2
    topic=$3
    nodes=$4
    threads=$5
    mh_step=$6
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
    if [ -z $mh_step ] ; then
        mh_step=1
    fi
    if [ -z "$dataset" ] ; then
        dataset=enwiki
    fi
    if [ -z "$runid" ] ; then
        runid=`date +%m%d%H%M%S`
    fi


    init

    # run experiments
    runwarplda $runid
fi

