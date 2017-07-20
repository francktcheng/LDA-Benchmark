# this is hdfs path to input dataset
#dataset="/enwik"
# dataset="/data/enwiki"
dataset="/enwiki"
harpbin=harp-app-noeval-1.0-SNAPSHOT.jar
# use the local hadoop installation
hadoopbin=/scratch/fg474admin/hadoop-2.6.0/bin/hadoop

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
threads=($1)
#
for topic in ${topics[*]}; do
for node in ${nodes[*]}; do
for thread in ${threads[*]}; do
    alpha=`echo "scale=4; 50/$topic" | bc`
    echo "init:set alpha to 50/K = $alpha"
    beta=0.01
    
    # <datapath> <topics> <alpha> <beta> <iternum> <timer_lowbound> <timer_highbound>  $nodes $thread $blockratio
    #Usage: edu.iu.lda.LDALauncher <doc dir> <num of topics> <alpha> <beta> <num of iterations> <min training percentage> <max training percentage> <num of mappers> <num of threads per worker> <schedule ratio> <memory (MB)> <work dir> <print model>

    $hadoopbin jar $homedir/bin/$harpbin edu.iu.lda.LDALauncher $dataset $topic $alpha $beta $2 $lbound $hbound $node $thread $blockratio 100000 /model/$appname/ false


done
done
done
