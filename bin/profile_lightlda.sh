nodes=(1)
mh_steps=(4)
topics=(1000)
threads=($1)
dataset="enwiki"

curdir=`pwd`
homedir=`dirname $curdir`

echo "run with dataset=$dataset, mh_step=$mh"

#
for topic in ${topics[*]}; do
for mh in ${mh_steps[*]}; do
for node in ${nodes[*]}; do
for thread in ${threads[*]}; do
    export LIGHTLDABIN=lightlda.noeval
    #export MV2_ENABLE_AFFINITY=0
    #$homedir/bin/runlightlda-dist.sh $dataset 10 $topic $node $thread $mh Profiling
    $homedir/bin/runlightlda.sh $dataset $2 $topic $node $thread $mh Profiling
done
done
done
done
