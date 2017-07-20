datasets=("enwiki")
nodes=(4)

curdir=`pwd`
homedir=`dirname $curdir`

# if [ ! -f cluster.ip ] ; then
# #echo 'create a new cluster.ip'
#     echo "`hostname`ib" >cluster.ip
# fi

#
topics=(1000)
threads=($1)
for topic in ${topics[*]}; do
for dataset in ${datasets[*]}; do
for node in ${nodes[*]}; do
for thread in ${threads[*]}; do
    export NOMADLDABIN=f+nomad-lda.noeval
    #runlightlda.sh <dataset> <iters> <topics> <nodes> <threads> <mf_steps> <runid>
    $homedir/bin/runnomadlda.sh $dataset $2 $topic $node $thread 1000 Profiling
done
done
done
done

