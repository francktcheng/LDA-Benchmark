VERSION=0.2
# CS Machines
CXX = g++
MPICXX = mpicxx

#Maverick
#CXX=icc
#MPICXX=mpicxx

DEBUGCXXFLAGS = -fPIC -O3 -fopenmp -std=c++11
#CXXFLAGS = -fPIC -O2 -march=native -DNDEBUG -fopenmp -std=c++11
CXXFLAGS = -fPIC -O3 -march=native -DNDEBUG -fopenmp -std=c++11

# TBB related paths
TBBROOT = tbb
#TBBLIB = -L${TBBROOT}/lib -ltbb -ltbbmalloc
TBBLIB = -ltbb -ltbbmalloc

#INCLUDEPATH = -I${TBBROOT}/include
LIBS = ${TBBLIB} -lrt -lm -lc

all: lda-converter f+nomad-lda splda

splda: splda.cpp splda.h
	${CXX} ${CXXFLAGS} ${INCLUDEPATH} splda.cpp -o splda 

f+nomad-lda: dist-lda-heap.h dist-lda-heap.cpp sparse_matrix.h petsc-reader.h 
	${MPICXX} ${CXXFLAGS} -o f+nomad-lda dist-lda-heap.cpp ${LIBS}

lda-converter: converter.cpp sparse_matrix.h
	${CXX} ${CXXFLAGS} -o lda-converter converter.cpp

tbb/lib:
	make -C tbb/ 

tar: moreclean
	cd ..; mv ${VERSION} nomad-lda-exp-${VERSION}; tar cvzf nomad-lda-exp-${VERSION}.tgz nomad-lda-exp-${VERSION}; mv nomad-lda-exp-${VERSION} ${VERSION}

clean:
	rm -rf lda-converter splda f+nomad-lda
	
moreclean: clean
	make -C tbb moreclean
	make -C data moreclean
