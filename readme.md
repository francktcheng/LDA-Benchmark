how-to run lda trainers
====================

shared sourcedir: /share/jproject/fg474/ldapaper/profiling

install step:

-  copy the sourcedir to LOCALDIR, here use rsync to keep synchronized as the files may be keeping updated
    
    $rsync -aP /share/jproject/fg474/ldapaper/profiling LOCALDIR

-  profiling/work is the working directory

    $cd work

-  run the lda trainers
    
    $../bin/profile_warplda.sh

    here is the log when running on j-30, it runs at the speed of about 12s/iteration  
    StartNoeval: K=1000, alpha=0.050000, beta=0.010000, niter=10  
    [2017-06-05 11:14:18] Iteration 0, 12.935029 s, 10706428.146041 tokens/thread/sec, 8 threads, log_likelihood (per token) 0.000000, total 0.000000e+00, word_likelihood 0.000000e+00
    
-  check for the scripts for more details

experiment settings:

    dataset:           enwiki
    trainers: warplda, lightlda, nomadlda, harplda
    topic number: K=1k
    threadnumber: 8, 32
    platform: juliet(72 cores)
    metric:   cache,  thread load balance?, synchronize overhead?
