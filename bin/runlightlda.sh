#!/bin/bash

bindir=`dirname $0`
homedir=`dirname $bindir`
dataroot=$homedir/data/
lightldabin=$LIGHTLDABIN
if [[ *"lightlda"* == "$LD_LIBRARY_PATH" ]] ; then
    echo '' 
else
    echo "export LD_LIBRARY_PATH=$homedir/src/lightlda/multiverso/third_party/lib/"
    export LD_LIBRARY_PATH=$homedir/src/lightlda/multiverso/third_party/lib/:$LD_LIBRARY_PATH
fi

#
# init
#
init()
{
    #set alpha=50/K
    alpha=`echo "scale=8; 50/$topic" | bc`
    echo "init:set alpha to 50/K = $alpha"

    case $dataset in
        enwiki)
            #8 slice
            beta=0.01
            num_vocabs=1000000
            max_doc=3800000
            data_cap=8500
            model_cap=5000
            delta_cap=5000
            alias_cap=800
            ;;
    
        enwiki-8)
            #8 slice
            beta=0.01
            num_vocabs=1000000
            max_doc=3800000
            data_cap=8500
            model_cap=20000
            delta_cap=20000
            alias_cap=512
            ;;
        enwiki-1)
            #slice=1
            beta=0.01
            num_vocabs=1000000
            max_doc=3800000
            data_cap=8500
            model_cap=20000
            delta_cap=20000
            alias_cap=80000
            ;;
 
        clueweb30b)
            beta=0.01
            num_vocabs=999933
            max_doc=77000000
            data_cap=10000
            model_cap=20000
            delta_cap=10000
            ;;
        bigram)
            #alpha=0.1
            beta=0.01
            num_vocabs=20000000
            max_doc=3900000
            data_cap=8000
            model_cap=600
            delta_cap=600
            alias_cap=600
            ;;
        nytimes)
            #alpha=0.1
            beta=0.01
            num_vocabs=101636
            max_doc=300000
            data_cap=8000
            model_cap=600
            delta_cap=600
            alias_cap=600
            ;;
        pubmed2m)
            #alpha=0.1
            beta=0.01
            num_vocabs=126591
            max_doc=2000010
            data_cap=1500
            model_cap=20000
            delta_cap=10000
            alias_cap=20000
            ;;
        pubmed)
            #alpha=0.1
            beta=0.01
            num_vocabs=144400
            max_doc=8300000
            data_cap=6200
 
            ;;
        *) echo "unkown dataset: $dataset"; help; exit 1;;
    esac

    datadir=$dataroot/$dataset/lightlda-$nodes/
}


runlightlda()
{
    logfile="lightlda_"$dataset"_t"$topic"_"$nodes"x"$threads"_i"$iter"_"$alpha"_"$beta"_"$mh_step"_$1.log"
        # run it
        cmd="$homedir/bin/$lightldabin -num_vocabs $num_vocabs -num_topics $topic -num_servers $nodes -num_iterations $iter -alpha $alpha -beta $beta -mh_steps $mh_step -num_local_workers $threads -num_blocks 1 -max_num_document $max_doc -input_dir $datadir -data_capacity $data_cap -model_capacity $model_cap -delta_capacity $delta_cap -alias_capacity $alias_cap "
        echo $cmd |tee $logfile
        $bindir/$lightldabin -num_vocabs $num_vocabs -num_topics $topic -num_servers $nodes -num_iterations $iter -alpha $alpha -beta $beta -mh_steps $mh_step -num_local_workers $threads -num_blocks 1 -max_num_document $max_doc -input_dir $datadir -data_capacity $data_cap -model_capacity $model_cap -delta_capacity $delta_cap -alias_capacity $alias_cap |tee -a $logfile
}

#
# main
#
if [ $# -eq 0 ]; then
    echo "runlightlda.sh <dataset> <iters> <topics> <nodes> <threads> <mf_steps> <runid>"
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
        dataset=nytimes
    fi
    if [ -z "$runid" ] ; then
        runid=`date +%m%d%H%M%S`
    fi

    init

    # run experiments
    runlightlda $runid
fi

