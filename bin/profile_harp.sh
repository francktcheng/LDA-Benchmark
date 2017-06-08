# this is hdfs path to input dataset
#dataset="/enwik"
dataset="/data/enwiki"
harpbin=harp-app-noeval-1.0-SNAPSHOT.jar

curdir=`pwd`
homedir=`dirname $curdir`

#notimer
lbound=100
hbound=100

#timer
lbound=40
hbound=80
blockratio=2

nodes=(1)
topics=(1000)
threads=(8)
#
for topic in ${topics[*]}; do
for node in ${nodes[*]}; do
for thread in ${threads[*]}; do
    alpha=`echo "scale=4; 50/$topic" | bc`
    echo "init:set alpha to 50/K = $alpha"
    beta=0.01
    
    # <datapath> <topics> <alpha> <beta> <iternum> <timer_lowbound> <timer_highbound>  $nodes $thread $blockratio
    #Usage: edu.iu.lda.LDALauncher <doc dir> <num of topics> <alpha> <beta> <num of iterations> <min training percentage> <max training percentage> <num of mappers> <num of threads per worker> <schedule ratio> <memory (MB)> <work dir> <print model>

    hadoop jar $homedir/bin/$harpbin edu.iu.lda.LDALauncher $dataset $topic $alpha $beta 10 $lbound $hbound $node $thread $blockratio 100000 /model/$appname/ false


done
done
done
