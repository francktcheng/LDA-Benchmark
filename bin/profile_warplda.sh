threads=(8 32)
datasets=("enwiki")

curdir=`pwd`
homedir=`dirname $curdir`

for dataset in ${datasets[*]}; do
for thread in ${threads[*]}; do
    #runxx.sh <dataset> <iters> <topics> <nodes> <threads> <mf_steps> <runid>
    $homedir/bin/runwarplda.sh $dataset 10 1000 1 $thread 4 Profiling
done
done

