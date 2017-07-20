#include "dist-lda-heap.h"

#define ROOT 0 // Root procid
#include <sys/time.h>
#include <time.h>

#ifdef VTUNE_PROF
// to write trigger file for vtune profiling
#include <iostream>
#include <fstream>
using namespace std;
#endif

enum {INIT, GO, EVAL, STOP}; // system.status

uint64_t timenow(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((uint64_t)(tv.tv_sec) * 1000000 + tv.tv_usec);
}


inline int get_procid() { // {{{
	int mpi_rank(-1);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	assert(mpi_rank >= 0);
	return int (mpi_rank);
} // }}}
inline int get_nr_procs() { // {{{
	int mpi_size(-1);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	assert(mpi_size >= 0);
	return int (mpi_size);
} // }}}
// Parse Command Line {{{
void exit_with_help() {
	// Only root print the message.
	if(get_procid() != ROOT) exit(1); 
	printf(
			"Usage: \n"
			"    export MV2_ENABLE_AFFINITY=0\n"
			"    mpiexec -n 4 ./f+nomad-lda [options] data_dir [model_filename]\n"
			"options:\n"
			"    -k nr_topics : set the number of topics (default 10)\n"    
			"    -a alpha*k : set the Dirichlet prior for topics (default 50)\n"    
			"    -b beta : set the Dirichlet prior for words (default 0.01)\n"    
			"    -d delay : set the token-pass delay (default 10)\n"
			"    -n nr_samplers : set the number of samplers (default 4)\n"    
			"    -t max_iter: set the number of iterations (default 5)\n"    
			"    -T time_interval: set the interval between evaluations (default 5 sec)\n"
			"    -m max_tokens_per_msg: set the max tokens per message (default 20 tokens)\n"
			"    -p do_predict: do prediction or not (default 0)\n"    
			"    -S nr_samples for perplexity computation (default 100)\n"
			"    -l sampler_schedule (default 2)\n"
			"    -L proc_schedule (default 2)\n"
			"        0 for load_balancing\n"
			"        1 for rand permutation\n"
			"        2 for cyclic\n"
			"        3 for pure rand\n"
			"    -q verbose: show information or not (default 0)\n"
		  );
	exit(1);
}

dist_lda_param_t parse_command_line(int argc, char **argv, char *train_src, char *test_src, char *model_file_name) {
	dist_lda_param_t param;
	param.nr_procs = get_nr_procs();
	train_src[0] = test_src[0] = 0;

	int i;
	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'k':
				param.k = atoi(argv[i]);
				break;

			case 'a':
				param.alpha = atof(argv[i]);
				break;

			case 'b':
				param.beta = atof(argv[i]);
				break;

			case 'd':
				param.delay = atoi(argv[i]);
				//param.delay = 20000;
				break;

			case 'n':
				param.samplers = atoi(argv[i]);
				break;

			case 'm':
				param.max_tokens_per_msg = atoi(argv[i]);
				param.max_tokens_per_msg = 100;
				break;

			case 't':
				param.maxiter = atoi(argv[i]);
				break;

			case 'T':
				//param.maxinneriter = atoi(argv[i]);
				param.time_interval = atof(argv[i]);
				break;

			case 'p':
				param.do_predict = atoi(argv[i]);
				break;

			case 'q':
				param.verbose = atoi(argv[i]);
				break;

			case 'S':
				param.perplexity_samples = atoi(argv[i]);
				break;

			case 'l':
				param.sampler_schedule = atoi(argv[i]);
				break; 
			case 'L':
				param.proc_schedule = atoi(argv[i]);
				break;

			default:
				fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
				break;
		}
	}

	if (param.do_predict != 0) 
		param.verbose = 1;

	param.alpha = param.alpha/(double)param.k;
	// determine filenames
	if(i>=argc)
		exit_with_help();

	//strcpy(input_file_name, argv[i]);

	char filename[1024];
	size_t nr_words, nr_docs, nnz, Z_len;
	sprintf(filename, "%s/meta", argv[i]);
	FILE *fp = fopen(filename, "r");
	if(fp == NULL) {
		fprintf(stderr, "Error: meta-file %s does not exist.\n", filename);
		exit(1);
	} 
	if(fscanf(fp, "%lu", &nr_words) != 1) {
		fprintf(stderr, "Error: corrupted meta in line 1 of %s\n", filename);
		exit(1);
	}
	if(fscanf(fp, "%lu %lu %lu %s", &nr_docs, &nnz, &Z_len, filename) != 4) {
		fprintf(stderr, "Error: corrupted meta in line 2 of meta\n");
		exit(1);
	}
	sprintf(train_src, "%s/%s.petsc", argv[i], filename);
	
	//check existence 
	if (access(train_src, F_OK|R_OK) != 0) {
		printf("petsc format does not exist. Converting it now.\n");
		char cmd[1024];
		sprintf(cmd, "./lda-converter %s", argv[i]);
		system(cmd);
	}
	if(fscanf(fp, "%lu %lu %lu %s", &nr_docs, &nnz, &Z_len, filename) != EOF)
		sprintf(test_src,"%s/%s.petsc", argv[i], filename);
	fclose(fp);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = argv[i]+ strlen(argv[i])-1;
		while (*p == '/') 
			*p-- = 0;
		p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
	return param;
} // }}}

// Shared Memory among threads {{{
struct comm_space_t {
	// data, param, and model
	lda_blocks_t &training_set, &test_set;
	dist_lda_param_t &param;
	lda_model_t &model;

	int sender_id, receiver_id, main_thread;
	atomic<int> owner_of_Nt; // the ownership of model.Nt, -1 means not in this proc
	unsigned global_seed; // shared by all procs
	unsigned local_seed; // sahred by all threads of this proc

	atomic<int> system_state; // INIT, GO, EVAL, STOP
	atomic<int> alive_samplers;
	atomic<size_t> alive_tokens; // sum over it of all procs = total_nomadic_tokens
	size_t total_nomadic_tokens; // should be model.nr_words+1
	std::vector<size_t> tokens_of_proc; // approx #tokens in each proc.
	std::vector<size_t> updates_of_word;
	std::vector<size_t> occurrence_of_word; // local occurrence of each word
	size_t nr_Nt_sent, nr_Nt_recv;
	size_t nr_msg_sent, nr_msg_recv;
	size_t updates_of_Nt;
	double starttime;

	std::vector<int> proc_of_word; // the ownership of each word

	// Thread-local variables and job queues for nomadic tokens
	mat_t local_Nt, delta_Nt;
	std::vector<htree_t<double>> heap_trees;
	std::vector<std::array<uint16_t,3>> sampler_seeds;
	std::vector<con_queue<size_t>> job_queues;
	con_queue<size_t> sender_queue;

	struct scheduler_t { // {{{
		std::vector<con_queue<size_t>> &job_queues;
		std::vector<size_t> &tokens_of_proc;
		dist_lda_param_t &param;
		unsigned &local_seed;
		size_t &total_nomadic_tokens;

		struct var_t {
			int age;
			unsigned seed;
			std::vector<int> perm;
			var_t():age(0){}
		};
		std::vector<var_t, callocator<var_t>> vars;
		scheduler_t(comm_space_t &space): param(space.param), local_seed(space.local_seed),
			job_queues(space.job_queues), tokens_of_proc(space.tokens_of_proc), 
			total_nomadic_tokens(space.total_nomadic_tokens) {
			vars.resize(space.model.nr_words);
			for(auto w = 0u; w < vars.size(); w++)
				vars[w].seed = w;
			if(param.sampler_schedule == dist_lda_param_t::rand_perm) {
				for(auto &var : vars) {
					auto &perm = var.perm;
					perm.resize(param.samplers);
					for(auto t=0u; t < perm.size(); t++)
						perm[t] = t;
				}
			}
		}
		int next_lightloading_sampler(int sampler_id) { // {{{
			auto next_sampler = sampler_id;
			auto minsize = total_nomadic_tokens; 
			for(auto s = 0; s < param.samplers; s++) {
				if(s != sampler_id and minsize > job_queues[s].unsafe_size()) {
					minsize = job_queues[s].unsafe_size();
					next_sampler = s;
				}
			}
			return next_sampler;
		} // }}}
		int next_lightloading_proc(int procid) { // {{{
			auto next_proc = procid;
			auto minsize = total_nomadic_tokens;
			//auto minsize = alive_tokens; //total_nomadic_tokens;
			for(auto i = 0; i < param.nr_procs; i++)  {
				if(i != procid and minsize > tokens_of_proc[i]) {
					minsize = tokens_of_proc[i];
					next_proc = i;
				}
			}
			return next_proc;
		} // }}}
		void reset(size_t word) {assert(word<vars.size()); vars[word].age = 0;}
		bool back_to_sender(size_t word) {return (param.nr_procs==1)? false : vars[word].age == param.samplers;}
		int next_sampler(size_t word, int cur_sampler_id) { // {{{
			auto &var = vars[word];
			auto sched = param.sampler_schedule;
			int next_sid;
			if(sched == dist_lda_param_t::load_balance) 
				next_sid = next_lightloading_sampler(cur_sampler_id);
			else if(sched == dist_lda_param_t::cyclic)
				next_sid = (cur_sampler_id+1) % param.samplers;
			else if(sched == dist_lda_param_t::pure_rand)
				next_sid = rand_r(&var.seed) % param.samplers;
			else if(sched == dist_lda_param_t::rand_perm) {
				if(var.age % param.samplers == 0) {
					auto nr_samplers = var.perm.size();
					for(size_t i = 0U; i < nr_samplers; i++) {
						size_t j = rand_r(&var.seed) % (nr_samplers-i) + i;
						std::swap(var.perm[i], var.perm[j]);
					}
				}
				next_sid = var.perm[var.age%param.samplers];
			}
			var.age += 1;
			return next_sid;
		} // }}}
		int next_proc(int cur_proc_id) { // {{{
			auto sched = param.proc_schedule;
			if(sched == dist_lda_param_t::load_balance)
				return next_lightloading_sampler(cur_proc_id);
			else if(sched == dist_lda_param_t::cyclic)
				return (cur_proc_id+1) % param.nr_procs;
			else if(sched == dist_lda_param_t::pure_rand 
					|| sched == dist_lda_param_t::rand_perm)
				return (cur_proc_id+(rand_r(&local_seed)%(param.nr_procs-1)+1)) % (param.nr_procs);
		} // }}}
	}; // }}}
	scheduler_t sched;

	comm_space_t(lda_blocks_t &training, lda_blocks_t &test, dist_lda_param_t &p, lda_model_t &m, unsigned seed = 0)
		: training_set(training), test_set(test), param(p), model(m), global_seed(seed), sched(*this) {
		auto procid = get_procid();
		auto nr_samplers = param.samplers;
		auto nr_procs = get_nr_procs();

		system_state = INIT;
		alive_samplers = 0;
		// Initiate the nomatic token for Nt
		if(procid == rand_r(&global_seed) % nr_procs) {
			alive_tokens = 1; owner_of_Nt = 0;
		} else {
			alive_tokens = 0; owner_of_Nt = -1;
		}
		total_nomadic_tokens = model.nr_words + 1;
		//XXX no Nt alive_tokens = 0; owner_of_Nt = -1; total_nomadic_tokens --;
		updates_of_word.resize(model.nr_words, 0);
		updates_of_Nt = 0;
		nr_Nt_sent = nr_Nt_recv = nr_msg_sent = nr_msg_recv = 0;

		if(nr_procs > 1) {
			sender_id = nr_samplers;
			receiver_id = nr_samplers + 1;
			main_thread = sender_id;
		} else {
			sender_id = -1;
			receiver_id = -1; main_thread = 0; }

		// Initial proc location of each nomadic token
		tokens_of_proc.resize(nr_procs);
		proc_of_word.resize(model.nr_words);
		for(auto &w: proc_of_word) {
			w = rand_r(&global_seed) % nr_procs;
			tokens_of_proc[w]++;
			if(w == procid) alive_tokens++;
		}
		occurrence_of_word.resize(model.nr_words);
		for(auto &training : training_set) {
			for(auto word = 0U; word < model.nr_words; word++)
				occurrence_of_word[word] += training.word_ptr[word+1] - training.word_ptr[word];
		}
		// Initialize local Nt and delta Nt
		local_Nt.resize(nr_samplers, model.dim);
		delta_Nt.resize(nr_samplers, model.dim);
		heap_trees.resize(nr_samplers);
		sampler_seeds.resize(nr_samplers);
		for(auto s = 0; s < nr_samplers; s++) {
			double betabar = param.beta*model.nr_words;
			auto &D = heap_trees[s];
			D.resize(model.dim);
			sampler_seeds[s][0] = (unsigned short) (s | nr_samplers);
			sampler_seeds[s][1] = (unsigned short) (s<<1 | nr_samplers);
			sampler_seeds[s][2] = (unsigned short) (s<<2 | nr_samplers);
			for(auto t = 0U; t < model.dim; t++) {
				local_Nt[s][t] = model.Nt[t];
				delta_Nt[s][t] = model.Nt[t];
				D[t] = param.beta/(model.Nt[t]+betabar);
			}
			D.init_dense();
		}

		// Initialize Job queues
		job_queues.resize(nr_samplers);
		std::vector<size_t> wp(model.nr_words); // word permutation
		for(size_t i = 0; i < model.nr_words; i++) wp[i] = i;
		for(size_t i = 0; i < model.nr_words; i++) {
			size_t j = rand_r(&local_seed) % (model.nr_words-i) + i;
			size_t tmp = wp[i]; wp[i] = wp[j]; wp[j] = tmp;
			if(proc_of_word[wp[i]] == procid)
				job_queues[rand_r(&local_seed)%nr_samplers].push(wp[i]);
		}
	}
	int next_lightloading_sampler(int sampler_id) {
		auto next_sampler = sampler_id;
		auto minsize = total_nomadic_tokens; 
		for(auto s = 0; s < param.samplers; s++) {
			if(s != sampler_id and minsize > job_queues[s].unsafe_size()) {
				minsize = job_queues[s].unsafe_size();
				next_sampler = s;
			}
		}
		return next_sampler;

	}
	int next_lightloading_proc(int procid) {
		auto next_proc = procid;
		//auto minsize = total_nomadic_tokens;
		auto minsize = alive_tokens; //total_nomadic_tokens;
		for(auto i = 0; i < param.nr_procs; i++)  {
			if(i != procid and minsize > tokens_of_proc[i]) {
				minsize = tokens_of_proc[i];
				next_proc = i;
			}
		}
		return next_proc;
	}
	bool check_stop() { return system_state == STOP; }
	bool check_timeout(int thread_id) {
		if(system_state != GO) return true;
		else if (thread_id != main_thread) return false;
		else if (omp_get_wtime()-starttime < param.time_interval) return false;
		system_state = EVAL;
		return true;
	}
	void start_running(int thread_id) {
		assert(thread_id == main_thread);
		system_state = GO;
		starttime = omp_get_wtime();
	}
	void stop_running(int thread_id) {
		assert(thread_id == main_thread);
		system_state = STOP;
	}
}; // }}}

typedef std::vector<unsigned char> StreamType;
struct msg_t{ //{{{
	comm_space_t &space;
	std::vector<size_t> queue;
	StreamType buf;
	enum {mpi_Nwt_tag=13, mpi_Nt_tag=17, mpi_eval_tag=19};
	enum type_t {sender, receiver} type;
	bool eval;


	msg_t(comm_space_t &space, type_t type, bool eval=false): space(space), type(type), eval(eval){}
	size_t size() {return queue.size();}
	bool empty(){return queue.size() == 0;}


	// Send/Recv for Nt {{{
	size_t send_Nt(int dest_procid) {
		assert(type == sender && "Wrong: receiver tries to send!");
		int mpi_tag = eval? mpi_eval_tag : mpi_Nt_tag;
		if(space.owner_of_Nt != space.sender_id) {
			fprintf(stderr, "Attempt to Send Nt without ownership!\n");
			return 0;
		} else {
			buf.clear();
			//serialize<vec_t>(space.model.Nt, buf);
			//memcpy(&buf[0], &space.model.Nt[0], buf.size());
			//MPI_Ssend(&buf[0], (int32_t) buf.size(), MPI_CHAR, dest_procid, mpi_Nt_tag, MPI_COMM_WORLD);
			MPI_Ssend(&space.model.Nt[0], (int32_t) sizeof(var_t)*space.model.dim, MPI_CHAR, dest_procid, mpi_tag, MPI_COMM_WORLD);
			return 1;
		}
	}
	bool recv_Nt(int src_procid = MPI_ANY_SOURCE) {
		assert(type == receiver && "Wrong: sender tries to recv!");
		int mpi_tag = eval? mpi_eval_tag : mpi_Nt_tag;
		if(space.owner_of_Nt != -1) { // Nt has been here!!
			return false;
		} else {
			int flag=0;
			MPI_Status status;
			MPI_Iprobe(MPI_ANY_SOURCE, mpi_tag, MPI_COMM_WORLD, &flag, &status);
			if(flag == 0) return false;
			src_procid = status.MPI_SOURCE;
			int buf_size = 0; MPI_Get_count(&status, MPI_CHAR, &buf_size);
			//buf.resize(buf_size);
			//MPI_Recv(&buf[0], buf_size, MPI_CHAR, src_procid, tag==-1?mpi_Nt_tag:tag, MPI_COMM_WORLD, &status);
			//space.model.Nt = std::move(deserialize<vec_t>(buf));
			MPI_Recv(&space.model.Nt[0], buf_size, MPI_CHAR, src_procid, mpi_tag, MPI_COMM_WORLD, &status);
			return true;
		}
	} // }}}

	// Send/Recv for Nwt {{{
	size_t send(int dest_procid) {
		assert(type == sender && "Wrong: receiver tries to send!");
		if(queue.size() == 0) return 0;
		else {
			size_t cnt = queue_to_buf();
			send_buf(dest_procid);
			return cnt;
		} 
	}

	void send_buf(int dest_procid) {
		int mpi_tag = eval? mpi_eval_tag : mpi_Nwt_tag;
		MPI_Ssend(&buf[0], (int32_t) buf.size(), MPI_CHAR, dest_procid, mpi_tag, MPI_COMM_WORLD);
	}
	size_t queue_to_buf() {
		size_t cnt = 0;
		buf.clear();
		buf.resize(sizeof(size_t)*(eval?1:2)+queue.size()*(sizeof(var_t)*space.model.dim+sizeof(size_t)));
		unsigned char *ptr = &buf[0];
		//serialize<size_t>(queue.size(), buf);
		size_t queue_size = queue.size(); memcpy(ptr, &queue_size, sizeof(size_t)); ptr+= sizeof(size_t);
		lda_model_t &model = space.model;
		for(auto word : queue) {
			size_t w = word; memcpy(ptr, &w, sizeof(size_t)); ptr+= sizeof(size_t);
			memcpy(ptr, &model.Nwt[word][0], sizeof(var_t)*model.dim); ptr+= sizeof(var_t)*model.dim;
			//serialize<size_t>(word, buf);
			//serialize<vec_t>(model.Nwt[word], buf);
			cnt ++;
		}
		//serialize<size_t>(space.alive_tokens-queue.size(), buf);
		if(not eval) {
			size_t loading = space.alive_tokens-queue.size(); 
			space.tokens_of_proc[get_procid()] = loading;
			memcpy(ptr, &loading, sizeof(size_t)); ptr+= sizeof(size_t);
		}
		assert((size_t)(ptr-&buf[0]) == buf.size());
		queue.clear();
		return cnt;
	}

	bool recv(int src_procid = MPI_ANY_SOURCE) {
		assert(type == receiver && "Wrong: sender tries to recv!");
		int mpi_tag = eval? mpi_eval_tag : mpi_Nwt_tag;
		int flag=0;
		MPI_Status status;
		MPI_Iprobe(src_procid, mpi_tag, MPI_COMM_WORLD, &flag, &status);
		if(flag == 0) return false;

		src_procid = status.MPI_SOURCE;
		int buf_size = 0; MPI_Get_count(&status, MPI_CHAR, &buf_size);
		buf.resize(buf_size);
		MPI_Recv(&buf[0], buf_size, MPI_CHAR, src_procid, mpi_tag, MPI_COMM_WORLD, &status);
		lda_model_t &model = space.model;
		//StreamType::const_iterator it = buf.begin(), end = buf.end();
		unsigned char *ptr = &buf[0];
		//size_t cnt = deserialize<size_t>(it, end);
		size_t cnt; memcpy(&cnt, ptr, sizeof(size_t)); ptr += sizeof(size_t); 
		for(auto i = 0U; i < cnt; i++) {
			//auto word = deserialize<size_t>(it, end);
			//model.Nwt[word] = std::move(deserialize<vec_t>(it, end));
			size_t word; memcpy(&word, ptr, sizeof(size_t)); ptr += sizeof(size_t);
			memcpy(&model.Nwt[word][0], ptr, sizeof(var_t)*model.dim); ptr += sizeof(var_t)*model.dim;
			queue.push_back(word);
		}
		//space.tokens_of_proc[src_procid] = deserialize<size_t>(it, end);
		if(not eval) {
			memcpy(&space.tokens_of_proc[src_procid], ptr, sizeof(size_t)); 
			ptr += sizeof(size_t);
		}
		assert((size_t)(ptr-&buf[0]) == buf.size());
		buf.clear();
		return true;
	}//}}}

	void push(size_t &word){ queue.push_back(word); }
	bool pop(size_t &word) { 
		if(queue.size() == 0) return false;
		word=queue.back(); queue.pop_back(); 
		return true;
	}
}; // }}}

void sampler_fun(int thread_id, comm_space_t &space) {//{{{
	auto &training = space.training_set[thread_id];
	auto &job_queue = space.job_queues[thread_id];
	auto &sched = space.sched;

	auto &Nwt = space.model.Nwt;
	auto &Ndt = space.model.Ndt;
	auto &Nt = space.model.Nt;
	auto &local_Nt = space.local_Nt[thread_id], &delta_Nt = space.delta_Nt[thread_id];
	
	auto dim = space.model.dim;
	auto alpha = space.param.alpha, beta = space.param.beta, betabar = beta*training.nr_words;
	auto delay = space.param.delay;

	auto *seed = &space.sampler_seeds[thread_id][0];
	unsigned int_seed = thread_id;

	htree_t<double> &D = space.heap_trees[thread_id]; // used to maintain (beta+Nw[t])/(delta_Nt[t]+betabar)

	bool alive = false;
	bool only_sampler = space.param.samplers*space.param.nr_procs == 1;

	std::vector<double> C(dim);
	size_t threshold = (size_t) floor(2.0*dim/(log2((double)dim)));
	auto check_ownership_of_Nt = [&] { // {{{
		if(space.owner_of_Nt == thread_id and not only_sampler) {
			if(delay > 0) {
				delay--;
			} else {
				for(auto t = 0U; t < dim; t++) {
					/*
					   Nt[t] += delta_Nt[t];
					   local_Nt[t] = Nt[t]; delta_Nt[t] = 0;
					   */
					Nt[t] += delta_Nt[t]-local_Nt[t];
					delta_Nt[t] = local_Nt[t] = Nt[t];
					D[t] = beta/(delta_Nt[t]+betabar);
				}
				D.init_dense();
				if(space.param.nr_procs == 1) {
					space.owner_of_Nt = (thread_id+1)%space.param.samplers;
				} else {
					space.owner_of_Nt = thread_id+1;
				}
				space.updates_of_Nt++;
				delay = space.param.delay;
			}
		}
	}; // }}}

	// Wait the GO signal to start running
	while(true) {
		// Finite State Machine
		if(space.system_state == INIT) { // {{{
			if(alive == false) {
				alive = true;
				space.alive_samplers++;
			}
			std::this_thread::yield(); continue;
		} else if (space.system_state == EVAL) {
			if(alive) { 
				space.alive_samplers--;
				alive = false;
			}
			std::this_thread::yield(); continue;
		} else if (space.system_state == STOP) {
			if(alive) {
				space.alive_samplers--;
				alive = false;
			}
			return;
		} else if(space.system_state == GO) {
			if(alive == false) {
				space.alive_samplers++;
				alive = true;
			}
		} // }}}

		if(thread_id == space.main_thread and space.check_timeout(thread_id)) {
			alive = false;
			space.alive_samplers --;
			return;
		}
		check_ownership_of_Nt();

		size_t cur_word;
		if(not job_queue.try_pop(cur_word)) {
			//std::this_thread::sleep_for(std::chrono::milliseconds(1));
			//printf("t %d empty\n", thread_id);
			std::this_thread::yield(); continue;
		}

		auto &Nw = Nwt[cur_word];
		if(training.word_ptr[cur_word] != training.word_ptr[cur_word+1]) { // {{{
			// Initialize D for cur_word
			if(Nw.nz_idx.size() > threshold){ // {{{
				for(auto t: Nw.nz_idx) 
					D.true_val[t] = (beta+Nw[t])/(Nt[t]+betabar);
				D.init_dense();
			} else  {
				for(auto t: Nw.nz_idx)
					D.set_value(t, (beta+Nw[t])/(Nt[t]+betabar));
			} // }}}

			for(auto idx = training.word_ptr[cur_word]; idx != training.word_ptr[cur_word+1]; idx++) {
				space.updates_of_word[cur_word] += training.val[idx];
				//check_ownership_of_Nt();

				auto cur_doc = training.doc_idx[idx];
				auto &Nd = Ndt[cur_doc];

				for(auto zidx = training.Z_ptr[idx]; zidx != training.Z_ptr[idx+1]; zidx++) {
					auto cur_topic = training.Zval[zidx];
					register double reg_denom = 1.0/((--delta_Nt[cur_topic])+betabar);
					--Nw[cur_topic]; // --Nd[cur_topic];
					double D_old = D.true_val[cur_topic];
					double D_new = reg_denom*(beta+Nw[cur_topic]);
					D.true_val[cur_topic] = D_new;
					bool is_D_updated = false;

					// Handle Inner Product (Part C) and locate Nd[cur_topic] {{{
					size_t nz_C = 0;
					double Csum = 0;
					doc_mat_t<var_t>::vec_t::element_t *ptr_Nd_cur_topic = NULL;
					for(auto &elem: Nd) {
						if(elem.idx == cur_topic) {
							elem.value--;
							ptr_Nd_cur_topic = &elem;
						} 
						C[nz_C++] = (Csum+=elem.value*D.true_val[elem.idx]);
					}
					/*
					size_t *ptr_Nd_cur_topic = NULL;
					for(auto &t : Nd.nz_idx) {
						if(Nd[t]) {
							C[nz_C++] = (Csum+= Nd[t]*D.true_val[t]);
						} else { 
							C[nz_C++] = Csum;
							ptr_Nd_cur_topic = &t;
						}
					}
					*/
					// }}}

					size_t new_topic = dim; // an invalid value to start with
					double Dsum = Csum + alpha*(D.total_sum()-D_old+D_new);
					double sample = erand48(seed)*Dsum;
					if(sample < Csum) { // {{{
						auto *ptr = C.data();
						new_topic = Nd[std::upper_bound(ptr, ptr+Nd.size(), sample)-ptr].idx;
						//new_topic = Nd.nz_idx[std::upper_bound(ptr, ptr+Nd.nz_idx.size(), sample)-ptr];
						//	nr_C++;
					} else {
						sample = (sample-Csum)/alpha;
						D.update_parent(cur_topic, D_new-D_old);
						is_D_updated = true;
						new_topic = D.log_sample(sample);
						//	nr_D++;
					} // }}}
					training.Zval[zidx] = new_topic;
					assert(new_topic < dim);

					// Add counts for the new_topic {{{
					reg_denom = 1.0/((++delta_Nt[new_topic])+betabar);
					++Nw[new_topic]; // ++Nd[new_topic]; 
					if(cur_topic != new_topic) {
						if(ptr_Nd_cur_topic->value==0) 
							Nd.remove_ptr(ptr_Nd_cur_topic);
						Nd.add_one(new_topic);
						if(Nw[cur_topic]==0) Nw.pop(cur_topic);
						if(Nw[new_topic]==1) Nw.push(new_topic);
						if(not is_D_updated) D.update_parent(cur_topic, D_new-D_old);
						D.set_value(new_topic, reg_denom*(beta+Nw[new_topic]));

					} else { // cur_topic == new_topic
						ptr_Nd_cur_topic->value++;
						if(is_D_updated) 
							D.set_value(cur_topic, D_old);
						else 
							D.true_val[cur_topic] = D_old;
					} // }}}
				}
			}
			// Reset D
			if(Nw.nz_idx.size() > threshold){ // {{{
				for(auto t: Nw.nz_idx)
					D.true_val[t] = beta/(Nt[t]+betabar);
				D.init_dense();
			} else  {
				for(auto t: Nw.nz_idx)
					D.set_value(t, beta/(Nt[t]+betabar));
			} // }}}
		} // }}}

#ifdef GG
		// pass this token to next thread
		if(space.param.nr_procs == 1) { // {{{
			if(space.param.samplers == 1) {
				job_queue.push(cur_word);
			} else {
				Nw.where++;
				/*
				auto schedule = space.param.sampler_schedule;
				int next_sampler;
				if(0) {
				} else if(schedule == dist_lda_param_t::load_balance) {
					next_sampler = space.next_lightloading_sampler(thread_id);
				} else if(schedule == dist_lda_param_t::rand_perm) {
					if(Nw.where >= space.param.samplers) {
						Nw.where = 0; Nw.shuffe(&int_seed);
					}
					next_sampler = Nw.get_sampler();
				} else if(schedule == dist_lda_param_t::cyclic) {
					next_sampler = Nw.where % space.param.samplers;
				} else if(schedule == dist_lda_param_t::pure_rand) {
					next_sampler = rand_r(&int_seed) % space.param.samplers;
				}
				space.job_queues[next_sampler].push(cur_word);
				*/

				if (space.param.sampler_schedule == dist_lda_param_t::load_balance) {
					auto next_sampler = space.next_lightloading_sampler(thread_id);
					space.job_queues[next_sampler].push(cur_word);
				} else if (space.param.sampler_schedule == dist_lda_param_t::rand_perm) {
					if (Nw.where >= space.param.samplers) {
						//if (next_sampler >= space.param.samplers) {
						Nw.where = 0;
						Nw.shuffle(&int_seed);
						job_queue.push(cur_word);
					} else {
						//	auto next_sampler = Nw.get_sampler();
						auto next_sampler = Nw.where % space.param.samplers;
						// auto next_sampler = rand() % space.param.samplers;
						space.job_queues[next_sampler].push(cur_word);
					}
				}
			} // }}}
		} else { // nr_procs > 1 {{{
			Nw.where++;
			if (space.param.sampler_schedule == dist_lda_param_t::load_balance) {
				if(Nw.where >= 1*space.param.samplers) {
					Nw.where = 0;
					space.sender_queue.push(cur_word);
				} else {
					auto next_sampler = space.next_lightloading_sampler(thread_id);
					space.job_queues[next_sampler].push(cur_word);
				}
			} else if (space.param.sampler_schedule == dist_lda_param_t::rand_perm) {
				//if (next_sampler >= space.param.samplers) {
				if (Nw.where >= 1*space.param.samplers) {
					Nw.where = 0;
					space.sender_queue.push(cur_word);
				} else {
					//auto next_sampler = Nw.get_sampler();
					auto next_sampler = Nw.where % space.param.samplers;
					space.job_queues[next_sampler].push(cur_word);
				}
			}
		} // }}}
#else
		if(only_sampler) {
			job_queue.push(cur_word);
		} else {
			if(space.param.nr_procs == 1) {
				auto next_sampler = sched.next_sampler(cur_word, thread_id);
				space.job_queues[next_sampler].push(cur_word);
			} else {
				if(sched.back_to_sender(cur_word)) {
					space.sender_queue.push(cur_word);
				} else {
					auto next_sampler = sched.next_sampler(cur_word, thread_id);
					space.job_queues[next_sampler].push(cur_word);
				}
			}
		}
#endif
	}
}//}}}

void sender_fun(int thread_id, comm_space_t &space) {//{{{
	assert(thread_id == space.sender_id && "Wrong sender ID");

	const size_t max_tokens_per_msg = space.param.max_tokens_per_msg;
	auto &sched = space.sched;
	msg_t msg(space, msg_t::sender);
	int procid = get_procid(); 
	int next_procid = (procid+1) % space.param.nr_procs;
	rng_type rng(thread_id);
#ifdef GG
	auto next_dest = [&]()-> int {
		return next_procid;
		return rand_r(&space.local_seed) % space.param.nr_procs;
		//return space.next_lightloading_proc(procid);
		if(space.param.proc_schedule == dist_lda_param_t::load_balance) {
			return space.next_lightloading_proc(procid);
		} else { // (space.param.proc_schedule == dist_lda_param_t::rand_perm) {
			return rand_r(&space.local_seed) % space.param.nr_procs;
		}
	} ;
#endif

	while(true) {
		if(space.system_state == INIT) continue;
		if(space.system_state != GO) break;
		if(space.check_timeout(thread_id)) break;

		if(space.owner_of_Nt == thread_id) {
			msg.send_Nt(next_procid);
			space.alive_tokens--;
			space.owner_of_Nt = -1;
			space.nr_Nt_sent++;
		}

		// reduce the frequent check_timeout() calls, which involve system calls.
		for(auto i = 0U; i < max_tokens_per_msg; i++) {
			size_t word;
			if(space.sender_queue.try_pop(word)) 
				msg.push(word);
		}

		//if(msg.size() >= max_tokens_per_msg) {
		if(msg.size() >= 1) {
#ifdef GG
			int dest_procid = next_dest(); 
#else
			int dest_procid = sched.next_proc(procid);
#endif
			auto token_sent = msg.send(dest_procid);
			space.alive_tokens -= token_sent;
			space.nr_msg_sent++;
		}
	}
#ifdef GG
	int dest_procid = next_dest();
#else
	int dest_procid = sched.next_proc(procid);
#endif
	space.alive_tokens -= msg.send(dest_procid);

}//}}}

void receiver_fun(int thread_id, comm_space_t &space) {//{{{
	assert(thread_id == space.receiver_id && "Wrong receiver ID");
	auto &sched = space.sched;
	msg_t msg(space, msg_t::receiver);
	mat_t &Nwt = space.model.Nwt;
	int samplers = space.param.samplers;
	int procid = get_procid();
	unsigned int_seed = procid*samplers+thread_id;

	while(true) {
		if(space.system_state == INIT) continue;
		if(space.system_state == STOP) break;
		if(msg.recv_Nt()) {
			space.alive_tokens++;
			space.owner_of_Nt = 0; // the first sampler
			space.nr_Nt_recv++;
		}
		if(msg.recv()) {
			space.nr_msg_recv++;
			size_t word;
			while(msg.pop(word)) {

				vec_t &Nw = Nwt[word];
				Nw.gen_nz_idx();
				if(samplers == 1) {
					space.job_queues[0].push(word);

					/*
				} else if (space.occurrence_of_word[word] == 0 and space.param.nr_procs > 1) {
					space.sender_queue.push(word);
					space.alive_tokens ++;
					space.proc_of_word[word] = procid;
				} else  if (space.alive_tokens * space.param.nr_procs >= 2*space.total_nomadic_tokens) {
					space.sender_queue.push(word);
					space.alive_tokens ++;
					space.proc_of_word[word] = procid;
					*/
				} else {
#ifdef GG
					if(space.param.sampler_schedule == dist_lda_param_t::load_balance) {
						Nw.where = 0;
						auto next_sampler = space.next_lightloading_sampler(thread_id);
						space.job_queues[next_sampler].push(word);

					} else if (space.param.sampler_schedule == dist_lda_param_t::rand_perm) {
						Nw.init_perm(samplers); Nw.shuffle(&int_seed);
						//space.job_queues[Nw.get_sampler()].push(word);
						space.job_queues[Nw.where].push(word);
					}
#else
					sched.reset(word);
					//auto next_sampler = sched.next_sampler(word, thread_id);
					auto next_sampler = sched.next_sampler(word, word % samplers);
					space.job_queues[next_sampler].push(word);
#endif
				}
				space.proc_of_word[word] = procid;
				space.alive_tokens++;
			}
		}
	}
}//}}}

double compute_training_LL(comm_space_t &space) { // {{{
	// unpack variables from space
	dist_lda_param_t &param = space.param;
	lda_model_t &model = space.model;
	lda_blocks_t &training = space.training_set;
	auto &Ndt = model.Ndt;
	auto &Nwt = model.Nwt;
	auto dim = model.dim;
	auto nr_words = model.nr_words;
	auto alpha = param.alpha, beta = param.beta;
	auto alphabar = alpha*dim, betabar = beta*model.nr_words;

	double LL = 0, localLL = 0;
//#pragma omp parallel for reduction(+:localLL)
//	for(auto doc = training.start_doc; doc < training.end_doc; doc++) {
//		auto &Nd = Ndt[doc];
//		var_t sum_Nd = 0;
//		double tmpLL = 0.0;
//		for(auto &elem: Nd) {
//			tmpLL += lgamma(alpha+elem.value) - lgamma(alpha);
//			sum_Nd += elem.value;
//		}
//		/*
//		for(auto t = 0U; t < dim; t++) {
//			if ( Nd[t] != 0 ) {
//				tmpLL += lgamma(alpha+ Nd[t]) - lgamma(alpha);
//				sum_Nd += Nd[t];
//			}
//		}
//		*/
//		tmpLL += lgamma(alphabar) - lgamma(alphabar + sum_Nd);
//		localLL += tmpLL;
//	}
//	MPI_Allreduce(&localLL, &LL, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	vec_t Nt(dim);
	size_t nonZeroTypeTopics = 0;
	//double model_LL = 0;
	for(auto word = 0U; word < nr_words; word++) {
		auto &Nw = Nwt[word];
		for(auto t = 0U; t < dim; t++) {
			if ( Nw[t] != 0) {
				nonZeroTypeTopics ++;
				LL += lgamma(beta+Nw[t]);
				Nt[t] += Nw[t];
			}
		}
	}

	size_t valid_topics = 0;
	for(auto t = 0U; t < dim; t++) 
		if ( Nt[t] != 0) {
			LL -= lgamma(betabar+Nt[t]);
			valid_topics ++;
		}

	LL += valid_topics*lgamma(betabar) - nonZeroTypeTopics*lgamma(beta);
	return LL;
}//}}}

double compute_perplexity(comm_space_t &space) {//{{{
	dist_lda_param_t &param = space.param;
	lda_model_t &model = space.model;
	lda_smat_t &test = space.test_set[0];
	mat_t &Nwt = model.Nwt;
	auto dim = model.dim;
	auto nr_words = model.nr_words;
	auto alpha = param.alpha, alphabar = alpha*dim;
	size_t nr_samples = param.perplexity_samples;

	// assume param is a cummulated sum
	auto cat_sample = [&](double_vec_t &param, unsigned short seed[3]) -> int{
		double randnum = erand48(seed) * param[dim-1];
		auto it = std::upper_bound(param.begin(), param.end(), randnum);
		return it - param.begin();
	};

	auto logsumexp = [&](double_vec_t& x) -> double{
		auto largest = *std::max_element(x.begin(), x.end());
		double result = 0;
		for(auto &v : x)
			result += exp(v - largest);
		return log(result)+largest;
	};

	std::vector<size_t> Nt(dim);
	for(auto word = 0U; word < nr_words; word++) {
		auto &Nw = Nwt[word];
		for(auto t = 0U; t < dim; t++)
			Nt[t] += Nw[t];
	}
	double_mat_t topic_dist_params(nr_words, double_vec_t(dim,0.0));
	for(auto word = 0U; word < nr_words; word++) {
		auto &Nw = Nwt[word];
		auto &param = topic_dist_params[word];
		for(auto t = 0U; t < dim; t++) {
			param[t] = (double)Nw[t]/(double)Nt[t];
			if(t > 0) param[t] += param[t-1];
		}
	}

	auto nr_threads = omp_get_max_threads();
	std::vector<size_t>totalw_pool(nr_threads);
	std::vector<double>perplexity_pool(nr_threads);
	std::vector<std::vector<size_t>>hist_pool(nr_threads, std::vector<size_t>(dim));
	std::vector<double_vec_t>logweights_pool(nr_threads, double_vec_t(nr_samples));

	std::vector<std::array<uint16_t,3>> seed_pool(nr_threads);
	for(auto s = 0; s < nr_threads; s++) {
		seed_pool[s][0] = (unsigned short) (s | nr_threads);
		seed_pool[s][1] = (unsigned short) (s<<1 | nr_threads);
		seed_pool[s][2] = (unsigned short) (s<<2 | nr_threads);
	}

	auto computer_doc_perplexity = [&](size_t doc) { // {{{
		auto thread_id = omp_get_thread_num();
		auto &totalw = totalw_pool[thread_id];
		auto &perplexity = perplexity_pool[thread_id];
		auto &hist = hist_pool[thread_id];
		auto &logweights = logweights_pool[thread_id];
		auto *seed = &seed_pool[thread_id][0];
		size_t cnt = 0;
		for(auto s = 0U; s < nr_samples; s++) {
			for(auto &t: hist) t = 0;
			double log_qq = 0.0, log_joint = 0.0;
			cnt = 0;
			for(auto idx = test.doc_ptr[doc]; idx != test.doc_ptr[doc+1]; idx++) {
				auto word = test.word_idx[idx];
				auto occurrence = test.val_t[idx];
				if(topic_dist_params[word][dim-1] == 0) continue;
				auto &topic_param = topic_dist_params[word];
				for(auto o = 0; o < occurrence; o++) {
					auto t = cat_sample(topic_param, seed);
					hist[t]++;
					double norm_const = topic_param[dim-1];
					double param_t = topic_param[t];
					if(t != 0) param_t -= topic_param[t-1];
					log_qq += log(param_t/norm_const);
					log_joint += log(param_t);
					cnt++;
				}
			}
			log_joint += lgamma(alphabar)-lgamma(alphabar+cnt);
			for(auto t = 0U; t < dim; t++)
				log_joint += lgamma(hist[t]+alpha)-lgamma(alpha);
			logweights[s] = log_joint - log_qq;
		}
		perplexity += logsumexp(logweights) - log(nr_samples);
		totalw += cnt;
	}; //}}}

	omp_set_num_threads(nr_threads);
#pragma omp parallel for
	for(auto doc = test.start_doc; doc < test.end_doc; doc++) 
		computer_doc_perplexity(doc);

	size_t totalw_local = 0, totalw = 0;
	for(auto &w: totalw_pool) totalw_local += w;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&totalw_local, &totalw, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

	double perplexity_local = 0.0, perplexity = 0.0;
	for(auto &p: perplexity_pool) perplexity_local += p;
	MPI_Allreduce(&perplexity_local, &perplexity, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	return exp(-perplexity/totalw);
}//}}}

// Distributed Collapsed Gibbs Sampler for LDA {{{
void dist_lda_CGS(lda_blocks_t &training_set, lda_blocks_t &test_set, dist_lda_param_t &param) {
	// main thread is also the sender thread
	int nr_samplers = param.samplers;
	int nr_procs = param.nr_procs;
	int procid = get_procid();
	size_t total_words = 0;

	// Model Initialization 
	test_set[0].initialize_Z_docwise(param.k, procid);
	if(procid==ROOT) printf("start model init\n");
	omp_set_num_threads(omp_get_num_procs());
#pragma omp parallel for
	for(auto s = 0; s < nr_samplers; s++)
		training_set[s].initialize_Z_wordwise(param.k, procid*nr_samplers+s);
	lda_model_t model(param, training_set, procid);
	auto dim = model.dim;

	if(procid==ROOT) printf("starting model init\n");
	model.initialize_with_blocks(training_set);
	if(procid==ROOT) printf("done model init\n");

	auto all_reduce_for_Nwt = [&] { // {{{
		// Allreduce for model.Nwt
		const unsigned int MAX_TOKEN_PER_MSG = 4096;
		vec_t send_buf(model.dim * MAX_TOKEN_PER_MSG), recv_buf(dim*MAX_TOKEN_PER_MSG);
		auto nr_words = model.nr_words;
		size_t cnt = 0;
		while (cnt + MAX_TOKEN_PER_MSG < nr_words) {
			for(size_t i = 0U; i < MAX_TOKEN_PER_MSG; i++)
				memcpy(&send_buf[i*dim], &model.Nwt[cnt+i][0], sizeof(var_t)*dim);
			MPI_Allreduce(&send_buf[0], &recv_buf[0], MAX_TOKEN_PER_MSG*dim, MPI_VAR_T, MPI_SUM, MPI_COMM_WORLD);
			for(size_t i = 0U; i < MAX_TOKEN_PER_MSG; i++)
				memcpy(&model.Nwt[cnt+i][0], &recv_buf[i*dim], sizeof(var_t)*dim);
			cnt+=MAX_TOKEN_PER_MSG;
		}
		auto remaining = nr_words - cnt;
		for(auto i = 0U; i < remaining; i++)
			memcpy(&send_buf[i*dim], &model.Nwt[cnt+i][0], sizeof(var_t)*dim);
		MPI_Allreduce(&send_buf[0], &recv_buf[0], remaining*dim, MPI_VAR_T, MPI_SUM, MPI_COMM_WORLD);
		for(auto i = 0U; i < remaining; i++)
			memcpy(&model.Nwt[cnt+i][0], &recv_buf[i*dim], sizeof(var_t)*dim);
	}; //}}}

	if(nr_procs > 1) { // Multiple Machine Case {{{

		all_reduce_for_Nwt();

		// Update model.Nt
		auto &N = model.Nt; for(auto &val : N) val = 0;
		for(auto word = 0U; word < model.nr_words; word++) {
			auto &Nw = model.Nwt[word];
			for(auto t = 0U; t < dim; t++)
				N[t] += Nw[t];
		}

		//if(procid == ROOT) { printf("init phase done! %ld tokens\n", total_words); }

	} // }}}
	model.construct_sparse_idx();
	total_words = 0;
	for(auto &nt : model.Nt)
		total_words += nt;
	if(procid == ROOT) { 
       time_t _t = time(0);
       char _str[64];
       strftime(_str, sizeof(_str), "%Y-%m-%d %H:%M:%S", localtime(&_t));
       //printf ("[%s] ", _str);
       //printf("init phase done! %ld tokens\n", total_words); 
       printf("[%s] init phase done! %ld tokens\n", _str, total_words); 
    }
	comm_space_t space(training_set, test_set, param, model);

	int thread_id = space.main_thread; 
	std::vector<std::thread> samplers;
	std::thread *receiver = NULL;
	if(param.nr_procs > 1) 
		receiver = new std::thread(receiver_fun, space.receiver_id, std::ref(space));
	msg_t eval_send_msg(space, msg_t::sender, true), eval_recv_msg(space, msg_t::receiver, true);

	if(param.nr_procs == 1){
		for(auto s = 1; s < nr_samplers; s++) {
			samplers.push_back(std::thread(sampler_fun, s, std::ref(space)));
		}
	} else {
		for(auto s = 0; s < nr_samplers; s++){
			samplers.push_back(std::thread(sampler_fun, s, std::ref(space)));
		}
	}
	double totaltime = 0.;
    double _elapse = 0.;
    size_t _lastnwt = 0;
    int _nodes = get_nr_procs(); 

	if(procid == ROOT) { 
       time_t _t = time(0);
       char _str[64];
       strftime(_str, sizeof(_str), "%Y-%m-%d %H:%M:%S", localtime(&_t));
       //printf ("[%s] ", _str);
       //printf("init phase done! %ld tokens\n", total_words); 
       printf("[%s] start training now\n", _str); 
    }

#ifdef VTUNE_PROF
    //write trigger file to disk and enable vtune
    ofstream vtune_trigger;
    vtune_trigger.open("vtune-flag.txt");
    vtune_trigger << "Start training process and trigger vtune profiling.\n";
    vtune_trigger.close();
#endif

	for(auto iter = 0; iter < param.maxiter; iter++) {
        uint64_t _start = timenow();
		if(param.nr_procs == 1) {
			space.start_running(thread_id);
			sampler_fun(thread_id, space);
		} else {
			space.start_running(thread_id);
			sender_fun(thread_id, space);
		}
		while(true) {if (space.alive_samplers == 0) break;}


		if(param.nr_procs > 1) { // Handle multi-machine cases {{{
			size_t global_tokens = 0;
			while(global_tokens != space.total_nomadic_tokens) {
				size_t local_tokens = space.alive_tokens;
				/*
				   MPI_Allgather(&local_tokens, 1, MPI_LONG_LONG, &space.tokens_of_proc[0], 1, MPI_LONG_LONG, MPI_COMM_WORLD);
				   global_tokens = 0;
				   for(auto &i : space.tokens_of_proc)
				   global_tokens += i;
				   */
				global_tokens = 0;
				MPI_Allreduce(&local_tokens, &global_tokens, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
			}
		}// }}}
		double timex = omp_get_wtime() - space.starttime;
		totaltime += timex;

#ifndef NDEBUG
		if(procid == ROOT) { printf("origin"); for(auto i : space.tokens_of_proc) printf("%ld ", i); puts(""); }
#endif


		std::vector<unsigned short> membership(model.nr_words);
		// Starting Evaluation...
		// Broadcast model.Nwt to evaybody.. //{{{
		for(auto &Q: space.job_queues) {
			auto size = Q.unsafe_size();
			auto Qcopy(Q);
			while(not Qcopy.empty()) {
				size_t word;
				Qcopy.try_pop(word);
				membership[word] = 1;
				//eval_send_msg.push(word);
			}
			assert(Q.unsafe_size() == size);
		}
		auto Q(space.sender_queue);
		while(not Q.empty()) {
			size_t word;
			Q.try_pop(word);
			membership[word] = 1;
			//eval_send_msg.push(word);
		}

        //evaluation part
		for(auto j = 0u; j < model.nr_words; j++) if(membership[j] == 0) {
			auto & Nw = model.Nwt[j];
			for(auto t = 0u; t < model.dim; t++)
				Nw[t] = 0;
		}

		all_reduce_for_Nwt();
		//eval_send_msg.queue_to_buf();
		//MPI_Barrier(MPI_COMM_WORLD);
		/*
		   for(auto i = 0; i < nr_procs; i++) {
		   if(i == procid) {
		   for(auto dest = 0; dest < nr_procs; dest++)
		   if(dest != i)
		   eval_send_msg.send_buf(dest);
		   } else {
		   while(not eval_recv_msg.recv(i)); 
		   }
		   MPI_Barrier(MPI_COMM_WORLD);
		   } 
		   */

		//}}}

        uint64_t _end1 = timenow();

		double train_LL = compute_training_LL(space);

		double test_perplexity = 0; 
		if(space.param.do_predict)
			test_perplexity = compute_perplexity(space);

		auto sum_vec = [&](std::vector<size_t>& v)->size_t { size_t ret = 0; for(auto &x:v) ret+= x; return ret; };

		size_t local_Nwt_cnt = sum_vec(space.updates_of_word), local_Nt_cnt = space.updates_of_Nt;
		size_t Nwt_cnt=0, Nt_cnt=0;
		MPI_Allreduce(&local_Nwt_cnt, &Nwt_cnt, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_Nt_cnt, &Nt_cnt, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        uint64_t _end2 = timenow();
        _elapse += (_end2 - _start)/1000000.0;

        //add the workers log for each node
		if(1) {
			printf("rank %d iter %d localNwt %ld localNt %ld", procid, iter+1, local_Nwt_cnt, local_Nt_cnt);
			puts("");
			fflush(stdout);
		}

		if(procid == ROOT) {
            //time_t rawtime;
            //struct tm * timeinfo;
            //time(&rawtime);
            //timeinfo = localtime(&rawtime);
            //printf ("%s:", asctime(timeinfo));
            time_t _t = time(0);
            char _str[64];
            strftime(_str, sizeof(_str), "%Y-%m-%d %H:%M:%S", localtime(&_t));
            printf ("[%s] ", _str);


			printf("iter %d time %.4g totaltime %.4g ", iter+1, timex, totaltime);
			printf("time-1 %.4g time-2 %.4g eplasetime %.4g ", (_end1 - _start)/1000000.0, (_end2 - _end1)/1000000.0, _elapse);
			printf("training-LL %.6g ", train_LL);
			if(param.do_predict)
				printf("perplexity %.6g ", test_perplexity);
			//printf("Nwt %ld avg %g Nt %ld", Nwt_cnt, (double)Nwt_cnt/(space.model.nr_words*space.param.nr_procs*space.param.samplers), Nt_cnt);
			printf("Nwt %ld avg %g Nt %ld ", Nwt_cnt, (double)Nwt_cnt/(double)total_words, Nt_cnt);
            double _time_periter =  (_end1 - _start)/1000000.0;
			printf("nxt %dx%d, throughput %.6e", _nodes, param.samplers, (double)(Nwt_cnt - _lastnwt)/param.samplers/_nodes/_time_periter);
            _lastnwt = Nwt_cnt;
			puts("");
			fflush(stdout);
		}
		std::vector<size_t>loading(space.param.nr_procs, 0);
		MPI_Allgather(&space.alive_tokens, 1, MPI_LONG_LONG, &loading[0], 1, MPI_LONG_LONG, MPI_COMM_WORLD);
		// Cheat // for(auto i = 0; i < space.param.nr_procs; i++) space.tokens_of_proc[i] = loading[i];

		if(procid == ROOT and get_nr_procs() > 1) { printf("after"); for(auto i =0; i < param.nr_procs; i++) printf("%ld/%ld ", space.tokens_of_proc[i], loading[i]); puts(""); }
#ifndef NDEBUG

		std::vector<size_t>nr_msg_sent(space.param.nr_procs, 0);
		MPI_Gather(&space.nr_msg_sent, 1, MPI_LONG_LONG, &nr_msg_sent[0], 1, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);
		std::vector<size_t>nr_msg_recv(space.param.nr_procs, 0);
		MPI_Gather(&space.nr_msg_recv, 1, MPI_LONG_LONG, &nr_msg_recv[0], 1, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);
		std::vector<size_t>nr_Nt_sent(space.param.nr_procs, 0);
		MPI_Gather(&space.nr_Nt_sent, 1, MPI_LONG_LONG, &nr_Nt_sent[0], 1, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);
		std::vector<size_t>nr_Nt_recv(space.param.nr_procs, 0);
		MPI_Gather(&space.nr_Nt_recv, 1, MPI_LONG_LONG, &nr_Nt_recv[0], 1, MPI_LONG_LONG, ROOT, MPI_COMM_WORLD);

		if(procid == ROOT) {
			//for(auto i = 0; i < loading.size(); i++) printf("proc %d %ld\n", i, loading[i]);
			for(auto i = 0u; i < loading.size(); i++)
				printf("proc %u %ld Nt: %ld %ld msg: %ld %ld\n", i, loading[i], 
						nr_Nt_sent[i], nr_Nt_recv[i], nr_msg_sent[i], nr_msg_recv[i]);
		}
#endif
		space.nr_msg_sent = space.nr_msg_recv = space.nr_Nt_sent = space.nr_Nt_recv = 0;


        //end of iter
        uint64_t _end3 =timenow();
        _elapse += (_end3 - _end2)/1000000.0;
	}

	space.stop_running(thread_id);
	for(auto &th : samplers)
		th.join();
	if(param.nr_procs > 1) { 
		receiver->join(); 
		delete receiver; 
	}
}// }}}

int main(int argc, char* argv[]){

	int mpi_thread_provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
	if(mpi_thread_provided != MPI_THREAD_MULTIPLE) {
		fprintf(stderr, "MPI multiple no provided!!\n");
		exit_with_help();
	}


	char model_file_name[1024], train_src[1024], test_src[1024];
	dist_lda_param_t param = parse_command_line(argc, argv, train_src, test_src, model_file_name);
	int procid = get_procid();

	if(procid == ROOT) {
		printf("options:\nmpiexec -n %d %s", get_nr_procs(), argv[0]);
		for(int i = 1; i < argc; i++)
			printf(" %s", argv[i]);
		puts("");
	}

	double loading_time = omp_get_wtime();
	// Distributed Data Loading
	lda_blocks_t training_set, test_set;
	petsc_reader_t training_reader(train_src);
	//doc_partitioner_t training_partitioner(training_reader.rows, param.nr_procs, param.samplers);
	doc_partitioner_t training_partitioner(training_reader, param.nr_procs, param.samplers);
	training_set.load_data(procid, training_reader, training_partitioner);
	training_reader.clear_space();

	petsc_reader_t test_reader(test_src);
	doc_partitioner_t test_partitioner(test_reader.rows, param.nr_procs, 1);
	test_set.load_data(procid, test_reader, test_partitioner);
	test_reader.clear_space();
	loading_time = omp_get_wtime() - loading_time;

	MPI_Barrier(MPI_COMM_WORLD);
	if(procid == ROOT) printf("Data loading done (%g sec)!\n", loading_time);

	dist_lda_CGS(training_set, test_set, param);

	MPI_Finalize();

	return 0;
}

void test(){

	/* testing I/O
	   for(auto &Zmat : training_set) Zmat.initialize_Z_wordwise(param.k);
	   size_t Zsum = 0;
	   for(auto i = 0; i < nr_samplers; i++)
	   Zsum += training_set[i].Z_len;
	   printf("procid %d: Zsum %ld nr_samplers %d\n", procid, Zsum, nr_samplers);

	   size_t tmp = 0;
	   MPI_Reduce(&Zsum, &tmp, 1, MPI_LONG, MPI_SUM, ROOT, MPI_COMM_WORLD);
	   if(procid == ROOT) {
	   printf("total -> %ld\n", tmp);
	   }
	   */
	/*
	   printf("procid %d nr_samplers %d\n", get_procid(), param.samplers);
	   token_pool_t<tc_t> pool(param.k, param.samplers, 5*param.samplers);
	   auto test = [&](int id) {
	   for(auto i =0; i < 4; i++) {
	   int32_t pid = pool.pop();
	   pool[pid].key = id;
	   }
	   };
	   std::vector<std::thread> tt;;
	   for(int i = 0; i < param.samplers; i++)
	   tt.push_back(std::thread(test, i));
	   for(auto &th : tt)
	   th.join();
	   for(auto i = 0; i < 4*param.samplers; i++)
	   printf("%d ", pool[i].key);
	   puts("");
	   printf("pool size %ld\n", pool.size());
	   */
}
