#ifndef LDA_MPI_H
#define LDA_MPIH

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <vector>
#include <array>
#include <tuple>
#include <random>
#include <memory>
#ifndef _GLIBCXX_USE_SCHED_YIELD
#define _GLIBCXX_USE_SCHED_YIELD
#endif
#ifndef _GLIBCXX_USE_NANOSLEEP
#define _GLIBCXX_USE_NANOSLEEP
#endif
#include <thread>
#include <cmath>
#include <chrono>
#include <functional> // for std::ref used for std::thread
#include <unistd.h> // access for existence
#include <omp.h>
#include <mpi.h>

#include "tbb/atomic.h"
#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_vector.h"
#include "tbb/scalable_allocator.h"
#include "tbb/cache_aligned_allocator.h"
#include "sparse_matrix.h"
#include "petsc-reader.h"

//#include "serialize.h"

typedef std::mt19937 rng_type;

template <typename T> 
//using sallocator = tbb::scalable_allocator<T>;
using sallocator = std::allocator<T>;

template <typename T> 
//using callocator = tbb::cache_aligned_allocator<T>;
using callocator = std::allocator<T>;

template <typename T>
using con_queue = tbb::concurrent_queue<T>;

template <typename T>
using atomic = tbb::atomic<T>;

class dist_lda_param_t {//{{{
	public:
		enum{load_balance=0, rand_perm=1, cyclic=2, pure_rand=3};

		long k;
		double alpha, beta;
		int delay;
		int samplers;
		int nr_procs;
		double time_interval;
		int proc_schedule;
		int sampler_schedule;
		int maxinneriter;
		int maxiter;
		int max_tokens_per_msg;
		int perplexity_samples;
		int lrate_method, num_blocks; 
		int do_predict, verbose;
		dist_lda_param_t() {
			k = 10;
			alpha = 50.0/k;
			beta = 0.01;
			delay = 10;
			samplers = 4;
			nr_procs = 1;
			maxinneriter = 5;
			time_interval = 5.0;
			//sampler_schedule = load_balance;
			//proc_schedule = load_balance;
			sampler_schedule = cyclic;
			proc_schedule = cyclic;
			maxiter = 5;
			max_tokens_per_msg = 20;
			perplexity_samples = 100;
			do_predict = 0;
			verbose = 0;
		}
};//}}}

typedef int wc_t; // type for word_count
typedef sparse_matrix<wc_t> smat_t; // sparse word count matrix
typedef smat_subset_iterator_t<wc_t> smat_subset_it; // subset iterator
typedef PETSc_reader<wc_t> petsc_reader_t; // PETSC_reader

// for topic-model
typedef long var_t; // variable type, size_t for topic_count, the main type used in the model

#if defined( RERM_64BIT )
	typedef int64_t IndexType;
#define MPIU_INT MPI_LONG_LONG_INT
#else
	typedef int IndexType;
#define MPIU_INT MPI_INT
#endif

#define MPI_VAR_T MPI_INT64_T

// divide docs into nr_procs*nr_samplers blocks.
class doc_partitioner_t: public std::vector<int> { //{{{
	public:
		size_t nr_docs, nr_blocks, blocksize;
		int nr_procs, nr_samplers;
		doc_partitioner_t(size_t nr_docs_ = 0, int nr_procs_ = 0, int nr_samplers_ = 0) {
			if(nr_procs_ != 0 and nr_samplers_ != 0)
				init(nr_docs_, nr_procs_, nr_samplers_);
		}
		doc_partitioner_t(petsc_reader_t &reader, int nr_procs_ = 0, int nr_samplers_ = 0) {
			if(nr_procs_ != 0 and nr_samplers_ != 0)
				init(reader, nr_procs_, nr_samplers_);
		}
		void init(size_t nr_docs_, int nr_procs_, int nr_samplers_) {
			nr_docs = nr_docs_; nr_procs = nr_procs_; nr_samplers = nr_samplers_;
			nr_blocks = nr_procs * nr_samplers;
			blocksize = nr_docs/nr_blocks + ((nr_docs%nr_blocks)?1:0);
			this->resize(nr_blocks+1);
			for(auto i = 1U; i < nr_blocks; i++)
				this->at(i) = this->at(i-1) + blocksize;
			this->at(nr_blocks) = nr_docs;
		}
		void init(petsc_reader_t &reader, int nr_procs_, int nr_samplers_) {
			nr_docs = reader.rows; nr_procs = nr_procs_; nr_samplers = nr_samplers_;
			nr_blocks = nr_procs * nr_samplers;
			this->resize(nr_blocks+1, 0);
			auto &nnz_doc = reader.nnz_row;
			size_t nr_tokens = 0;
			for(size_t d = 0; d < nr_docs; d++) 
				nr_tokens += nnz_doc[d];
			size_t avg_tokens_per_block = nr_tokens/nr_blocks + 1;
			int bid = 1;;
			nr_tokens = 0;
			for(size_t d = 0; d < nr_docs; d++) {
				nr_tokens += nnz_doc[d];
				this->at(bid) += 1;
				if(nr_tokens > avg_tokens_per_block) {
					nr_tokens = 0;
					bid++;
				}
			}
			for(auto i = 1u; i <= nr_blocks; i++)
				this->at(i) += this->at(i-1);
			if(this->at(nr_blocks)!= nr_docs) puts("wrong");
		}
		size_t start_doc(int procid, int samplerid) {
			return this->at(procid*nr_samplers + samplerid);
		}
		size_t end_doc(int procid, int samplerid) {
			return this->at(procid*nr_samplers+samplerid+1);
		}
};//}}}

class lda_smat_t { // {{{
	public:
		smat_t word_count;
		size_t nr_docs, nr_words, start_doc, end_doc, nnz, Z_len;
		size_t *word_ptr, *doc_ptr, *Z_ptr;
		unsigned *doc_idx, *word_idx, *Zval;
		wc_t *val, *val_t;

		//size_t* &col_ptr; //=word_ptr;
		//unsigned* &row_idx; //=doc_idx;

		bool mem_alloc_by_me;

		lda_smat_t(){
			nr_docs = nr_words = start_doc = end_doc = Z_len = 0;
			doc_ptr = word_ptr = Z_ptr = NULL;
			word_idx = doc_idx = Zval = NULL;
			mem_alloc_by_me = false;
			//col_ptr = word_ptr; row_idx = doc_idx;
		}
		lda_smat_t(const lda_smat_t &Z){*this=Z; mem_alloc_by_me=false;}

		~lda_smat_t() { clear_space(); }

		void load_data(size_t nr_docs_, size_t nr_words_, size_t nnz_, 
				const char *filename, smat_t::format_t file_fmt) {
			word_count.load(nr_docs_, nr_words_, nnz_, filename, file_fmt);
			fill_attributes();
		}

		void load_data(smat_subset_it it) {
			word_count.load_from_iterator(it.get_rows(), it.get_cols(), it.get_nnz(), &it);
			fill_attributes();
		}

		void load_data(petsc_reader_t &reader, size_t start_row, size_t end_row, std::vector<int> *col_perm=NULL) {
			reader.receive_rows(start_row, end_row, word_count, col_perm);
			fill_attributes();
			start_doc = start_row; end_doc = end_row;
		}

		void fill_attributes() {
			nr_docs = word_count.rows; nr_words = word_count.cols; nnz = word_count.nnz;
			start_doc = 0; end_doc = nr_docs;
			word_ptr = word_count.col_ptr;
			doc_idx = word_count.row_idx;
			val = word_count.val;

			doc_ptr = word_count.row_ptr;
			word_idx = word_count.col_idx;
			val_t = word_count.val_t;
		}

		void initialize_Z_wordwise(int k, unsigned seed = 0) {
			Z_ptr = sallocator<size_t>().allocate(word_count.nnz+1);
			if(Z_ptr == NULL) {
				puts("No enough memory for Zptr\n");
			}
			Z_len = 0; Z_ptr[0] = 0;
			for(size_t idx = 0; idx < word_count.nnz; idx++) {
				Z_len += word_count.val[idx]; // we want CSC ordering
				Z_ptr[idx+1] = Z_len; 
			}
			Zval = sallocator<unsigned>().allocate(Z_len);
			if(Zval == NULL) {
				puts("No enough memory for Zval\n");
			}
			for(size_t zidx = 0; zidx < Z_len; zidx++)
				Zval[zidx] = rand_r(&seed) % k;
			mem_alloc_by_me = true;
		}

		void initialize_Z_docwise(int k, unsigned seed = 0) {
			Z_ptr = sallocator<size_t>().allocate(word_count.nnz+1);
			Z_len = 0; Z_ptr[0] = 0;
			for(size_t idx = 0; idx < word_count.nnz; idx++) {
				Z_len += word_count.val_t[idx]; // we wat CSR ordering
				Z_ptr[idx+1] = Z_len; 
			}
			Zval = sallocator<unsigned>().allocate(Z_len);
			for(size_t zidx = 0; zidx < Z_len; zidx++)
				Zval[zidx] = rand_r(&seed) % k;
			mem_alloc_by_me = true;
		}

		void clear_space(){
			if(mem_alloc_by_me) {
				if(Z_ptr) sallocator<size_t>().deallocate(Z_ptr, word_count.nnz+1);
				if(Zval) sallocator<unsigned>().deallocate(Zval, Z_len);
				//if(Z_ptr) free(Z_ptr); //sallocator<size_t>().deallocate(Z_ptr, word_count.nnz+1);
				//if(Zval) free(Zval); //sallocator<unsigned>().deallocate(Zval, Z_len);
			}
		}

};//}}}

class lda_blocks_t: public std::vector<lda_smat_t, callocator<lda_smat_t> >{ //{{{
	public:
		size_t nr_docs, nr_words, nr_blocks, start_doc, end_doc;
		std::vector<unsigned> block_of_doc;
		lda_blocks_t(){}

		// for multicore loading
		void load_data(size_t nr_blocks_, smat_t &wc) {
			auto range = [](size_t begin, size_t end)->std::vector<unsigned>{
				std::vector<unsigned> ret(end-begin);
				for(size_t i = 0; i < ret.size(); i++) ret[i] = begin+i;
				return ret;
			};

			nr_docs = wc.rows; nr_words = wc.cols; nr_blocks = nr_blocks_;
			start_doc = 0; end_doc = nr_docs;
			this->resize(nr_blocks);
			block_of_doc.resize(nr_docs);
			auto blocksize = nr_docs/nr_blocks + ((nr_docs%nr_blocks)?1:0); 
#pragma omp parallel for
			for(auto t = 0U; t < nr_blocks; t++) {
				const std::vector<unsigned> &subset = range(t*blocksize, std::min((t+1)*blocksize, nr_docs));
				this->at(t).load_data(wc.row_subset_it(subset));
				for(auto it:subset) block_of_doc[it] = t;
			}
		}

		// for distributed loading 
		void load_data(int procid, petsc_reader_t &reader, doc_partitioner_t &partitioner) {
			nr_docs = reader.rows;
			nr_words = reader.cols;
			nr_blocks = partitioner.nr_samplers;
			start_doc = partitioner.start_doc(procid, 0);
			end_doc = partitioner.end_doc(procid, nr_blocks-1);
			this->resize(nr_blocks);
#pragma omp parallel for
			for(auto samplerid = 0U; samplerid < nr_blocks; samplerid++) {
				size_t start_doc = partitioner.start_doc(procid, samplerid);
				size_t end_doc = partitioner.end_doc(procid, samplerid);
				this->at(samplerid).load_data(reader, start_doc, end_doc);
			}
		}
}; // }}}

// nonzero entry for spvec_t
template<typename val_type, typename idx_t=unsigned>
struct entry_t{ // {{{
	idx_t idx;
	val_type value;
	entry_t(idx_t  idx_=0, val_type value_=val_type()): idx(idx_), value(value_){}
	bool operator<(const entry_t& other) const { return (idx < other.idx);}
	bool operator==(const entry_t& other) const {return idx == other.idx;}
}; // }}}
template<typename val_type, typename alloc=callocator<val_type>>
struct base_vec_t : public std::vector<val_type, alloc> {//{{{
	typedef typename std::vector<val_type, alloc> super;
	base_vec_t(size_t size_=0): super(size_) {}
	base_vec_t(size_t size_, val_type &val_): super(size_, val_) {}

#ifdef GG
	int where; // the position in sampler_perm
	std::vector<int, callocator<int> > sampler_perm;
	void init_perm(int samplers) {
		where = 0;
		return; // XXX no permutation at all
		if(samplers > 0 and sampler_perm.size() == 0){
			sampler_perm.resize(samplers);
			for(auto i = 0; i < samplers; i++)
				sampler_perm[i] = i;
		}
	}
	void shuffle(unsigned *seed) { 
		auto nr_samplers = sampler_perm.size();
		for(size_t i = 0U; i < nr_samplers; i++) {
			size_t j = rand_r(seed) % (nr_samplers-i) + i;
			std::swap(sampler_perm[i], sampler_perm[j]);
		}
		//shuffle(sampler_perm.begin(), sampler_perm.end(), rng); 
	}
	int get_sampler() {
		if (where < (int)sampler_perm.size())
			return sampler_perm[where];
		else 
			return sampler_perm[where%(int)sampler_perm.size()];
	}
#endif
}; // }}}
template<typename val_type, typename idx_type, typename alloc=callocator<val_type>>
struct spidx_t : public base_vec_t<val_type, alloc>{ // {{{
	typedef typename ::base_vec_t<val_type, alloc> super;
	typedef idx_type idx_t;
	std::vector<idx_t, callocator<idx_t>> nz_idx;
	spidx_t(size_t size_=0): super(size_) {}
	spidx_t(size_t size_, val_type &val_): super(size_, val_){}
	void gen_nz_idx(int capacity=-1) { // {{{
		nz_idx.clear();
		if(capacity==-1) capacity = (int) this->size();
		nz_idx.reserve(capacity);
		for(auto t = 0U; t < this->size(); t++) 
			if((*this).at(t) > 0)
				nz_idx.push_back((idx_t)t);
	}
	bool check_idx() {
		size_t i = 0;
		for(auto t = 0U; t < this->size(); t++) {
			if((*this)[t]>0) {
				if(t != nz_idx[i]) return false;
				i++;
			}
		}
		return i==nz_idx.size();
	} // }}}
	void push(idx_t idx) { // {{{
		size_t len = nz_idx.size();
		if(len)  {
			nz_idx.push_back(idx); // nz_idx.size() = len+1 now.
			idx_t *start = nz_idx.data(), *end=start+len;
			idx_t *ptr = std::lower_bound(start, end, idx);
			if(ptr != end) {
				memmove(ptr+1, ptr, (len - (ptr-start))*sizeof(idx_t));
				*ptr = idx;
			}
		} else {
			nz_idx.push_back(idx);
		}
	} // }}}
	void pop(idx_t idx) { // {{{
		idx_t *start = nz_idx.data(), *end = start+nz_idx.size();
		idx_t *ptr = std::lower_bound(start, end, idx);
		if(ptr != end)
			remove_ptr(ptr);
	} // }}}
	void remove_ptr(idx_t *ptr) { // {{{
		size_t cnt = nz_idx.size()-(ptr+1-nz_idx.data());
		if(cnt) 
			memmove(ptr, ptr+1, cnt*sizeof(idx_t));
		nz_idx.pop_back();
	} // }}}
}; //}}}

template <typename val_type, typename idx_t, typename alloc=callocator<entry_t<val_type, idx_t>>>
struct spvec_t : public std::vector<entry_t<val_type, idx_t>, alloc>{ // {{{
	typedef typename std::vector<entry_t<val_type, idx_t>, alloc> super;
	typedef entry_t<val_type, idx_t> element_t;
	typedef std::vector<val_type> dvec_t;
	size_t len;
	spvec_t(size_t size_=0): len(size_), super(0) {}
	spvec_t(size_t size_, val_type &val_): len(size_), super(0) {}
	void init_with(const dvec_t& x, size_t capacity=0) {
		this->clear(); this->reserve(capacity);
		for(idx_t t = 0u; t < x.size(); t++)
			if(x[t] != 0) 
				this->push_back(element_t(t, x[t]));
	}
	void to_dvec_t(dvec_t &x) {
		for(auto &elem : *this) 
			x[elem.idx] = elem.value;
	}
	element_t *get_ptr(idx_t idx) { // {{{
		size_t nz = this->size();
		if(nz) {
			auto *start = this->data(), *end = start+nz;
			auto *ptr = std::lower_bound(start, end, idx);
			return ptr == end? ptr : NULL;
		} else 
			return NULL;
	} // }}}
	void add_one(idx_t idx, val_type val=val_type(1u)) { // {{{
		size_t nz = this->size();
		if(nz) {
			element_t *start = this->data(), *end = start+nz;
			element_t *ptr = std::lower_bound(start, end, idx);
			if(ptr != end) {
				if(ptr->idx == idx) 
					ptr->value++;
				else {
					ptrdiff_t pos = ptr - start;
					this->push_back(element_t(idx, val));
					start = this->data(); ptr = start+pos; 
					memmove(ptr+1, ptr, (nz - pos)*sizeof(element_t));
					*ptr = element_t(idx, val);
				}
			} else 
				this->push_back(element_t(idx, val));
		} else 
			this->push_back(element_t(idx, val));
	} // }}}
	void remove_ptr(element_t *ptr) { // {{{
		size_t cnt = this->size()-(ptr+1-this->data());
		if(cnt) 
			memmove(ptr, ptr+1, cnt*sizeof(element_t));
		this->pop_back();
	} // }}}
}; // }}}
template <typename val_type>
//using model_vec_t = base_vec_t<val_type>;
//using model_vec_t = spidx_t<val_type, size_t, std::allocator<val_type>>;
using model_vec_t = spidx_t<val_type, unsigned>;


template<typename val_type>
struct model_mat_t{ //{{{
	typedef model_vec_t<val_type> vec_t;
	size_t rows, cols, start_row, end_row;
	std::vector<vec_t, sallocator<vec_t> > buf;
	model_mat_t(size_t rows_=0, size_t cols_=0, size_t start_row_=0, size_t end_row_=0, int samplers=1) {
		resize(rows_, cols_, start_row_, end_row_, samplers);
	}
	vec_t& operator[](size_t id) {return buf[id-start_row];}
	const vec_t& operator[](size_t id) const {return buf[id-start_row];}
	void resize(size_t rows_=0, size_t cols_=0, size_t start_row_=0, size_t end_row_=0, int samplers=1) {
		if(end_row_ <= start_row_) end_row_ = rows_;
		rows = rows_; cols = cols_; start_row = start_row_; end_row = end_row_;
		buf.clear();
		buf.resize(end_row-start_row, vec_t(cols));
		if(buf.size() != end_row-start_row) {
			printf("No enough memory for model\n");
		}
#ifdef GG
		if(samplers > 1) 
			for(auto &vec: buf) 
				vec.init_perm(samplers);
#endif
	}
	void push_back(const vec_t &val, int samplers = 0) {
		buf.push_back(val); 
#ifdef GG
		buf[buf.size()-1].init_perm(samplers);
#endif
		end_row++;
	}

}; //}}}

template <typename val_type>
using doc_vec_t = spvec_t<val_type, unsigned>;

template<typename val_type>
struct doc_mat_t{ //{{{
	typedef doc_vec_t<val_type> vec_t;
	size_t rows, cols, start_row, end_row;
	std::vector<vec_t, sallocator<vec_t> > buf;
	doc_mat_t(size_t rows_=0, size_t cols_=0, size_t start_row_=0, size_t end_row_=0, int samplers=1) {
		resize(rows_, cols_, start_row_, end_row_, samplers);
	}
	vec_t& operator[](size_t id) {return buf[id-start_row];}
	const vec_t& operator[](size_t id) const {return buf[id-start_row];}
	void resize(size_t rows_=0, size_t cols_=0, size_t start_row_=0, size_t end_row_=0, int samplers=1) {
		if(end_row_ <= start_row_) end_row_ = rows_;
		rows = rows_; cols = cols_; start_row = start_row_; end_row = end_row_;
		buf.clear();
		buf.resize(end_row-start_row, vec_t(cols));
		if(buf.size() != end_row-start_row) {
			printf("No enough memory for model\n");
		}
	}
	void push_back(const vec_t &val, int samplers = 0) {
		buf.push_back(val); 
		end_row++;
	}

}; //}}}

// LDA Model {{{
typedef model_vec_t<var_t> vec_t;
typedef model_mat_t<var_t> mat_t;
typedef std::vector<atomic<var_t>, callocator<atomic<var_t> > > atom_vec_t;
typedef std::vector<double, callocator<double>> double_vec_t;
typedef std::vector<double_vec_t, callocator<double_vec_t>> double_mat_t;

struct lda_model_t {
	size_t dim; // #topics
	size_t nr_docs, nr_words;
	mat_t Nwt;
	doc_mat_t<var_t> Ndt;
	vec_t Nt;

	lda_model_t(size_t dim_=0, size_t nr_docs_ =0, size_t nr_words_=0) {
		dim=dim_;nr_docs=nr_docs_;nr_words=nr_words_;
		Ndt.resize(nr_docs,dim);
		Nwt.resize(nr_words,dim);
		Nt.resize(dim);
	}
	lda_model_t(dist_lda_param_t &param, lda_blocks_t &blocks, int procid = 0) {
		dim = param.k; nr_docs = blocks.nr_docs; nr_words = blocks.nr_words;
		Nwt.resize(nr_words, dim, 0, nr_words, blocks.nr_blocks);
		Ndt.resize(nr_docs, dim, blocks.start_doc, blocks.end_doc);
		Nt.resize(dim);
	}

	void check(int signal) {
		for(auto i = 0U; i < nr_words; i++)
			for(auto t = 0U; t < dim; t++)
				if(Nwt[i][t] < 0) {
					printf("Wrong at %d\n", signal);
					return;
				}
	}
	void initialize_with_blocks(lda_blocks_t &blocks) {
		for(size_t i = 0; i < blocks.size(); i++)
			initialize_with_blocks(blocks[i]);
	}

	void initialize_with_blocks(lda_smat_t &block) {
		std::vector<size_t> nz_doc(block.end_doc - block.start_doc);
		size_t tt = 0;
		for(size_t word = 0U; word < block.nr_words; word++) {
			for(auto idx = block.word_ptr[word]; idx != block.word_ptr[word+1]; idx++) {
				auto doc = block.doc_idx[idx];
				for(auto zidx = block.Z_ptr[idx]; zidx != block.Z_ptr[idx+1]; zidx++) {
					int topic = block.Zval[zidx];
					nz_doc[doc-block.start_doc]++;
					Nwt[word][topic]++;
					Nt[topic]++;
				}
				tt += block.val_t[idx];
			}
		}
		for(auto doc = block.start_doc; doc < block.end_doc; doc++)
			Ndt[doc].reserve(nz_doc[doc-block.start_doc]);
		for(size_t word = 0U; word < block.nr_words; word++) {
			for(auto idx = block.word_ptr[word]; idx != block.word_ptr[word+1]; idx++) {
				auto doc = block.doc_idx[idx];
				for(auto zidx = block.Z_ptr[idx]; zidx != block.Z_ptr[idx+1]; zidx++)
					Ndt[doc].add_one(block.Zval[zidx]);
			}
		}
	}
	void construct_sparse_idx() {
		omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for
		for(auto word=Nwt.start_row; word < Nwt.end_row; word++)
			Nwt[word].gen_nz_idx();
	}
}; // }}} 

// LDA old Model {{{
struct lda_model_t_old {
	size_t dim; // #topics
	size_t nr_docs, nr_words;
	mat_t Ndt, Nwt;
	vec_t Nt;

	lda_model_t_old(size_t dim_=0, size_t nr_docs_ =0, size_t nr_words_=0) {
		dim=dim_;nr_docs=nr_docs_;nr_words=nr_words_;
		Ndt.resize(nr_docs,dim);
		Nwt.resize(nr_words,dim);
		Nt.resize(dim);
	}
	lda_model_t_old(dist_lda_param_t &param, lda_blocks_t &blocks, int procid = 0) {
		dim = param.k; nr_docs = blocks.nr_docs; nr_words = blocks.nr_words;
		Nwt.resize(nr_words, dim, 0, nr_words, blocks.nr_blocks);
		Ndt.resize(nr_docs, dim, blocks.start_doc, blocks.end_doc);
		Nt.resize(dim);
	}

	void check(int signal) {
		for(auto i = 0U; i < nr_words; i++)
			for(auto t = 0U; t < dim; t++)
				if(Nwt[i][t] < 0) {
					printf("Wrong at %d\n", signal);
					return;
				}
	}
	void initialize_with_blocks(lda_blocks_t &blocks) {
		for(auto &block: blocks) 
			initialize_with_blocks(block);
	}

	void initialize_with_blocks(lda_smat_t &block) {
		for(size_t word = 0U; word < block.nr_words; word++) {
			for(auto idx = block.word_ptr[word]; idx != block.word_ptr[word+1]; idx++) {
				auto doc = block.doc_idx[idx];
				for(auto zidx = block.Z_ptr[idx]; zidx != block.Z_ptr[idx+1]; zidx++) {
					int topic = block.Zval[zidx];
					Ndt[doc][topic]++;
					Nwt[word][topic]++;
					Nt[topic]++;
				}
			}
		}
	}
	void construct_sparse_idx() {
		for(auto word=Nwt.start_row; word < Nwt.end_row; word++)
			Nwt[word].gen_nz_idx();
		for(auto doc=Ndt.start_row; doc < Ndt.end_row; doc++) {
			size_t nz = 0;
			for(auto &v: Ndt[doc]) 
				nz+=v;
			Ndt[doc].gen_nz_idx(nz);
		}
	}
}; // }}} 

template<typename T>
struct circular_queue_t { // {{{
	size_t head, tail;
	size_t capacity;
	T* buf;
	circular_queue_t(size_t capacity = 0): head(0), tail(0), capacity(capacity+1), buf(NULL){buf = (T*)malloc(sizeof(T)*(capacity));}
	~circular_queue_t(){clear_buf();}
	bool empty() {return head==tail;}
	T front() {return buf[head];}
	void pop() {head=(head+1)%capacity;}
	void push(const T& m) {buf[tail] = m; tail=(tail+1)%capacity;}
	void clear() {head=tail=0;}
	void clear_buf() {if(buf) free(buf);}
	void reserve(size_t cap) {capacity=cap+1; buf = (T*)realloc(buf, sizeof(T)*(capacity));}
}; // }}}

template<typename val_type=double>
struct htree_t{ // {{{
	size_t size;     // real size of valid elements
	size_t elements; // 2^ceil(log2(size)) capacity
	std::vector<val_type> val;
	double *true_val;
	//circular_queue_t<long> Q;
	size_t init_time, update_time, sample_time;

	double& operator[](size_t idx) { assert(idx < elements); return true_val[idx]; }
	const double& operator[] (size_t idx) const { assert(idx < elements); return true_val[idx]; }
	double get_init_time() {return (double)init_time/CLOCKS_PER_SEC;}
	double get_update_time() {return (double)update_time/CLOCKS_PER_SEC;}
	double get_sample_time() {return (double)sample_time/CLOCKS_PER_SEC;}
	void init_dense() { // {{{
		/*
		for(size_t pos = (elements+size)>>1; pos > 0; --pos)
			val[pos] = val[pos<<1] + val[(pos<<1)+1];
			*/
		for(size_t pos = elements-1; pos > 0; --pos) 
			val[pos] = val[pos<<1] + val[(pos<<1)+1];
	} // }}}
//	void init_sparse(size_t *nz_idx, size_t nz) { // {{{
//		//std::clock_t start_time = clock();
//		//init_dense(); 
//		//init_sparse_bfs(nz_idx, nz);
//		//init_sparse_dfs(nz_idx, nz);
//		init_sparse_new(nz_idx, nz);
//		//init_time += clock()-start_time;
//	}
//	void clear_sparse(size_t *nz_idx, size_t nz) { 
//		//std::clock_t start_time = clock();
//		//clear(); 
//		//clear_sparse_dfs(nz_idx, nz);
//		//clear_sparse_bfs(nz_idx, nz);
//		clear_sparse_new(nz_idx, nz);
//		//init_time += clock()-start_time;
//	}
//
//	void init_sparse_new(size_t *nz_idx, size_t nz) {
//		size_t start = (elements+nz_idx[0]) >> 1;
//		size_t end = (elements+nz_idx[nz-1]) >> 1;
//		while(end) {
//			for(size_t pos = start; pos <= end; pos++) {
//				size_t tmp = pos<<1;
//				val[pos] = val[tmp]+val[tmp+1];
//			}
//			start >>= 1;
//			end >>= 1;
//		}
//	}
//	void init_sparse_bfs(size_t *nz_idx, size_t nz) {
//		for(size_t i = 0; i < nz; ++i) 
//			Q.push((elements+nz_idx[i])>>1);
//		long last_pos = -1;
//		while(not Q.empty()) {
//			long pos = Q.front(); Q.pop();
//			if(pos != last_pos) {
//				val[pos] = val[pos<<1] + val[(pos<<1)+1];
//				last_pos = pos;
//				if(pos>>1) Q.push(pos>>1);
//			}
//		}
//	}
//	void init_sparse_dfs(size_t *nz_idx, size_t nz) {
//		for(size_t i = 0; i < nz; ++i) {
//			size_t idx = (nz_idx[i]+elements)>>1;
//			while(idx) {
//				val[idx] = val[idx<<1]+val[(idx<<1)+1];
//				idx >>= 1;
//			}
//		}
//	}
//
//	void clear_sparse_new(size_t *nz_idx, size_t nz) {
//		size_t end = elements+nz_idx[nz-1];
//		size_t start = elements+nz_idx[0];
//		while(end) {
//			for(size_t i = start; i <= end; i++)
//				val[i] = 0;
//			end >>= 1;
//			start >>= 1;
//		}
//	}
//	void clear_sparse_dfs(size_t *nz_idx, size_t nz) {
//		for(size_t i = 0; i < nz; i++) {
//			size_t pos = elements+nz_idx[i];
//			while(pos and val[pos]) {
//				val[pos] = 0;
//				pos >>= 1;
//			}
//		}
//	}
//	void clear_sparse_bfs(size_t *nz_idx, size_t nz) {
//		for(size_t i = 0; i < nz; ++i) {
//			size_t pos = elements+nz_idx[i];
//			if(pos>>1) Q.push(pos>>1);
//		}
//		while(not Q.empty()) {
//			long pos = Q.front(); Q.pop();
//			val[pos] = 0;
//			if(pos>>1) Q.push(pos>>1);
//		}
//	} // }}}
	void update_parent(size_t idx, val_type delta) { // {{{
		idx = (idx+elements)>>1;
		while(idx) {
			val[idx] += delta;
			idx >>= 1;
		}
	} // }}}
	void set_value(size_t idx, val_type value) { // {{{
		value -= val[idx+=elements]; // delta update
		while(idx) {
			val[idx] += value;
			idx >>= 1;
		}
	} // }}}
	// urnd: uniformly random number between [0,1]
	size_t log_sample(double urnd) { // {{{
		//urnd *= val[1]; 
		size_t pos = 1;
		while(pos < elements) {
			pos <<= 1; 
			if(urnd > val[pos]) 
				urnd -= val[pos++];
			/*
			double tmp = urnd - val[pos];
			if(tmp >= 0) {
				urnd = tmp;
				pos++;
			}
			*/
			/*
			if(urnd < val[pos*2]) 
				pos = pos*2;
			else { 
				urnd -= val[pos*2]; 
				pos = pos*2+1; 
			}
			*/
		}
		return pos-elements;
	} // }}}
	size_t linear_sample(double urnd) { // {{{
		//urnd = urnd*val[1];
		size_t pos = elements;
		while(urnd > 0)
			urnd -= val[pos++];
		if(pos >= elements+size) pos = elements+size-1;
		return pos-elements;
	} // }}}
	double total_sum() { return val[1]; }
	double left_cumsum(size_t idx) { // {{{
		if(idx == elements) return val[1];
		size_t pos = elements+idx+1;  
		double sum = 0;
		while(pos>1) {
			if(pos & 1)
				sum += val[pos^1];
			pos >>= 1;
		}
		return sum;
	} // }}}
	double right_cumsum(size_t idx) {return val[1] - left_cumsum(idx-1);}
	htree_t(size_t size=0): init_time(0),update_time(0),sample_time(0) { resize(size); }
	void resize(size_t size_ = 0) { // {{{
		size = size_;
		if(size == 0) val.resize(0);
		elements = 1;
		while(elements < size) elements *= 2;
		val.clear(); val.resize(2*elements, 0.0);
		true_val = &val[elements];
		//Q.reserve(elements); 
	} //}}}
	void clear() { for(auto &v: val) v = 0; }
}; // }}}

template <typename val_type>
struct token_pool_t{ // {{{
	typedef model_vec_t<val_type> token_t;
	int dim, nr_samplers;
	model_mat_t<val_type> pool;
	con_queue<int> avail_queue;
	token_pool_t(int dim, int nr_samplers, size_t init_size = 1024) {
		dim = dim; nr_samplers = nr_samplers;
		//pool.resize(init_size, token_t<val_type>(dim, nr_samplers));
		pool.resize(init_size, dim, 0, init_size, nr_samplers);
		for(auto pool_id = 0; pool_id < pool.size(); pool_id++) 
			avail_queue.push(pool_id);
	}
	token_t& operator[](int32_t pool_id) {return pool[pool_id];}
	const token_t& operator[](int32_t pool_id) const {return pool[pool_id];}
	size_t size() {return avail_queue.unsafe_size();}
	void push(int32_t pool_id) { avail_queue.push(pool_id); }
	int32_t pop() {
		int32_t pool_id = -1;
		bool succeed = avail_queue.try_pop(pool_id);
		if(succeed) return pool_id;
		else {
			pool_id = (int) pool.size();
			pool.push_back(token_t(dim), nr_samplers);
			return pool_id;
		}
	}
}; //}}}

#endif // LDA_MPI_H
