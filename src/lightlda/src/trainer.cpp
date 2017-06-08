#include "trainer.h"

#include "alias_table.h"
#include "common.h"
#include "data_block.h"
#include "eval.h"
#include "meta.h"
#include "sampler.h"
#include "model.h"

#include <multiverso/barrier.h>
#include <multiverso/stop_watch.h>
#include <multiverso/log.h>

namespace multiverso { namespace lightlda
{
    std::mutex Trainer::mutex_;
    double Trainer::doc_llh_ = 0.0;
    double Trainer::word_llh_ = 0.0;

    Trainer::Trainer(AliasTable* alias_table, 
		Barrier* barrier, Meta* meta) : 
        alias_(alias_table), barrier_(barrier), meta_(meta),
        model_(nullptr)
    {
        sampler_ = new LightDocSampler();
        model_ = new PSModel(this);
    }

    Trainer::~Trainer()
    {
        delete sampler_;
        delete model_;
    }

    void Trainer::TrainIteration(DataBlockBase* data_block)
    {
        StopWatch watch; watch.Start();
        LDADataBlock* lda_data_block =
            reinterpret_cast<LDADataBlock*>(data_block);

        DataBlock& data = lda_data_block->data();
        int32_t block = lda_data_block->block();
        int32_t slice = lda_data_block->slice();
        int32_t iter = lda_data_block->iteration();
        const LocalVocab& local_vocab = data.meta();
        
        int32_t id = TrainerId();
        int32_t trainer_num = TrainerCount();
        int32_t lastword = local_vocab.LastWord(slice);
        if (id == 0)
        {
            Log::Info("Rank = %d, Iter = %d, Block = %d, Slice = %d\n",
                Multiverso::ProcessRank(), lda_data_block->iteration(),
                lda_data_block->block(), lda_data_block->slice());
        }
        // Build Alias table
        if (id == 0) alias_->Init(meta_->alias_index(block, slice));
        barrier_->Wait();
        for (const int32_t* pword = local_vocab.begin(slice) + id;
            pword < local_vocab.end(slice);
            pword += trainer_num)
        {
            alias_->Build(*pword, model_);
        }
        if (id == 0) alias_->Build(-1, model_);
        barrier_->Wait();

        if (TrainerId() == 0)
        {
            Log::Info("Rank = %d, Alias Time used: %.2f s \n",
                Multiverso::ProcessRank(), watch.ElapsedSeconds());
        }
        int32_t num_token = 0;
        watch.Restart();
        // Train with lightlda sampler
        for (int32_t doc_id = id; doc_id < data.Size(); doc_id += trainer_num)
        {
            Document* doc = data.GetOneDoc(doc_id);
            num_token += sampler_->SampleOneDoc(doc, slice, lastword, model_, alias_);
        }

        //Barrier Fix
        //barrier_->Wait();

        //add threadlog
        //Log::Info("Rank = %d, Threadid = %d, Training Time used: %.2f s \n", 
        //        Multiverso::ProcessRank(), TrainerId(), watch.ElapsedSeconds());

        if (TrainerId() == 0)
        {
            Log::Info("Rank = %d, Training Time used: %.2f s \n", 
                Multiverso::ProcessRank(), watch.ElapsedSeconds());
            Log::Info("Rank = %d, sampling throughput: %.6f (tokens/thread/sec) \n", 
                Multiverso::ProcessRank(), double(num_token) / watch.ElapsedSeconds());
        }
        watch.Restart();
        // Evaluate loss function
        // Evaluate(lda_data_block);
        
        //if (iter % 5 == 0)
        if (iter == Config::num_iterations - 1)
        {
            Evaluate(lda_data_block);
            if (TrainerId() == 0)
                Log::Info("Rank = %d, Evaluation Time used: %.2f s \n",
                    Multiverso::ProcessRank(), watch.ElapsedSeconds());
        }
        // if (iter != 0 && iter % 50 == 0) Dump(iter, lda_data_block);

        // Clear the thread information in alias table
        if (iter == Config::num_iterations - 1) alias_->Clear();
    }

    void Trainer::Evaluate(LDADataBlock* lda_data_block)
    {
        double thread_doc = 0, thread_word = 0;

        DataBlock& data = lda_data_block->data();
        int32_t block = lda_data_block->block();
        int32_t slice = lda_data_block->slice();
        const LocalVocab& local_vocab = data.meta();

        // 1. Evaluate doc likelihood
        //for (int32_t doc_id = TrainerId(); doc_id < data.Size() && slice == 0;
        //    doc_id += TrainerCount())
        //{
        //    thread_doc += Eval::ComputeOneDocLLH(data.GetOneDoc(doc_id),
        //        sampler_->doc_topic_counter());
        //}
        //{
        //    std::lock_guard<std::mutex> lock(mutex_);
        //    doc_llh_ += thread_doc;
        //}
        //if (slice == 0 && barrier_->Wait())
        //{
        //    Log::Info("Rank = %d, doc likelihood : %e\n",Multiverso::ProcessRank(), doc_llh_);
        //}

        // 2. Evaluate word likelihood
        for (const int32_t* word = local_vocab.begin(slice) + TrainerId();
            word < local_vocab.end(slice) && block == 0; word += TrainerCount())
        {
            thread_word += Eval::ComputeOneWordLLH(*word, this);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            word_llh_ += thread_word;
        }
        if (block == 0 && barrier_->Wait())
        {
            Log::Info("Rank = %d, word likelihood : %e\n",  Multiverso::ProcessRank(),   word_llh_);
        }

        // 3. Evaluate normalize item for word likelihood
        if (TrainerId() == 0 && block == 0)
        {
            //Log::Info("Normalized likelihood : %e\n",
            //    Eval::NormalizeWordLLH(this));

            double _log_lh = Eval::NormalizeWordLLH(this);
            //add a total output

            Log::Info("Rank = %d, log_likelihood : %e\n",Multiverso::ProcessRank(), _log_lh + doc_llh_ + word_llh_);
            Log::Info("Rank = %d, word_log_likelihood : %e\n",Multiverso::ProcessRank(), _log_lh + word_llh_);

        }
        barrier_->Wait();
        doc_llh_ = 0;
        word_llh_ = 0;
    }

    void Trainer::Dump(int32_t iter, LDADataBlock* lda_data_block)
    {
        DataBlock& data = lda_data_block->data();
        int32_t slice = lda_data_block->slice();
        const LocalVocab& local_vocab = data.meta();

        std::string out = "model." + std::to_string(iter) + "." 
            + std::to_string(slice) + "." + std::to_string(TrainerId());
        std::ofstream fout(out);

        for (const int32_t* p = local_vocab.begin(slice) + TrainerId();
            p < local_vocab.end(slice); p += TrainerCount())
        {
            int word = *p;
            Row<int32_t>& row = GetRow<int32_t>(kWordTopicTable, word);
            Row<int32_t>::iterator iter = row.Iterator();

            fout << word;
            while (iter.HasNext())
            {
                int32_t topic = iter.Key();
                int32_t count = iter.Value();
                fout << " " << topic << ":" << count;
                iter.Next();
            }
            fout << std::endl;
        }

        fout.close();
    }

    void ParamLoader::ParseAndRequest(DataBlockBase* data_block)
    {
        LDADataBlock* lda_data_block =
            reinterpret_cast<LDADataBlock*>(data_block);
        // Request word-topic-table
        int32_t slice = lda_data_block->slice();
        DataBlock& data = lda_data_block->data();
        const LocalVocab& local_vocab = data.meta();

        for (const int32_t* p = local_vocab.begin(slice);
            p != local_vocab.end(slice); ++p)
        {
            RequestRow(kWordTopicTable, *p);
        }
        Log::Debug("Request params. start = %d, end = %d\n",
            *local_vocab.begin(slice), *(local_vocab.end(slice) - 1));
        // Request summary-row
        RequestTable(kSummaryRow);
    }
} // namespace lightlda
} // namespace multiverso
