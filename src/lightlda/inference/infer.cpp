#include "common.h"
#include "alias_table.h"
#include "data_stream.h"
#include "data_block.h"
#include "document.h"
#include "meta.h"
#include "util.h"
#include "model.h"
#include "inferer.h"
#include <vector>
#include <iostream>
#include <thread>
// #include <pthread.h>
#include <multiverso/barrier.h>

namespace multiverso { namespace lightlda
{     
    class Infer
    {
    public:
        static void Run(int argc, char** argv)
        {
            Log::ResetLogFile("LightLDA_infer." + std::to_string(clock()) + ".log");
            Config::Init(argc, argv);
            //init meta
            meta.Init();
            //init model
            LocalModel* model = new LocalModel(&meta); model->Init();
            //init document stream
            data_stream = CreateDataStream();
            //init documents
            InitDocument();
            //init alias table
            AliasTable* alias_table = new AliasTable();
            //init inferers
            std::vector<Inferer*> inferers;
            Barrier barrier(Config::num_local_workers);
            // pthread_barrier_t barrier;
            // pthread_barrier_init(&barrier, nullptr, Config::num_local_workers);
            for (int32_t i = 0; i < Config::num_local_workers; ++i)
            {
               inferers.push_back(new Inferer(alias_table, data_stream, 
                    &meta, model, 
                    &barrier, i, Config::num_local_workers));
            }

            //do inference in muti-threads
            Inference(inferers);

            //dump doc topic
            DumpDocTopic();
            
            //recycle space
            for (auto& inferer : inferers)
            {
                delete inferer;
                inferer = nullptr;
            }
            // pthread_barrier_destroy(&barrier);
            delete data_stream;
            delete alias_table;
            delete model;
        }
    private:
        static void Inference(std::vector<Inferer*>& inferers)
        {
            //pthread_t * threads = new pthread_t[Config::num_local_workers];
            //if(nullptr == threads)
            //{
            //    Log::Fatal("failed to allocate space for worker threads");
            //}
            std::vector<std::thread> threads;
            for(int32_t i = 0; i < Config::num_local_workers; ++i)
            {
                threads.push_back(std::thread(&InferenceThread, inferers[i]));
                //if(pthread_create(threads + i, nullptr, InferenceThread, inferers[i]))
                //{
                //    Log::Fatal("failed to create worker threads");
                //}
            }
            for(int32_t i = 0; i < Config::num_local_workers; ++i)
            {
                // pthread_join(threads[i], nullptr);
                threads[i].join();
            }
            // delete [] threads;
        }

        static void* InferenceThread(void* arg)
        {
            Inferer* inferer = (Inferer*)arg;
            // inference corpus block by block
            for (int32_t block = 0; block < Config::num_blocks; ++block)
            {
                inferer->BeforeIteration(block);
                for (int32_t i = 0; i < Config::num_iterations; ++i)
                {
                    inferer->DoIteration(i);
                }
                inferer->EndIteration();
            }
            return nullptr;
        }

        static void InitDocument()
        {
            xorshift_rng rng;
            for (int32_t block = 0; block < Config::num_blocks; ++block)
            {
                data_stream->BeforeDataAccess();
                DataBlock& data_block = data_stream->CurrDataBlock();
                int32_t num_slice = meta.local_vocab(block).num_slice();
                for (int32_t slice = 0; slice < num_slice; ++slice)
                {
                    for (int32_t i = 0; i < data_block.Size(); ++i)
                    {
                        Document* doc = data_block.GetOneDoc(i);
                        int32_t& cursor = doc->Cursor();
                        if (slice == 0) cursor = 0;
                        int32_t last_word = meta.local_vocab(block).LastWord(slice);
                        for (; cursor < doc->Size(); ++cursor)
                        {
                            if (doc->Word(cursor) > last_word) break;
                            // Init the latent variable
                            if (!Config::warm_start)
                                doc->SetTopic(cursor, rng.rand_k(Config::num_topics));
                        }
                    }
                }
                data_stream->EndDataAccess();
            }
        }


        static void DumpDocTopic()
        {
            Row<int32_t> doc_topic_counter(0, Format::Sparse, kMaxDocLength); 
            for (int32_t block = 0; block < Config::num_blocks; ++block)
            {
                std::ofstream fout("doc_topic." + std::to_string(block));
                data_stream->BeforeDataAccess();
                DataBlock& data_block = data_stream->CurrDataBlock();
                for (int i = 0; i < data_block.Size(); ++i)
                {
                    Document* doc = data_block.GetOneDoc(i);
                    doc_topic_counter.Clear();
                    doc->GetDocTopicVector(doc_topic_counter);
                    fout << i << " ";  // doc id
                    Row<int32_t>::iterator iter = doc_topic_counter.Iterator();
                    while (iter.HasNext())
                    {
                        fout << " " << iter.Key() << ":" << iter.Value();
                        iter.Next();
                    }
                    fout << std::endl;
                }
                data_stream->EndDataAccess();
            }
        }
    private:
        /*! \brief training data access */
        static IDataStream* data_stream;
        /*! \brief training data meta information */
        static Meta meta;
    };
    IDataStream* Infer::data_stream = nullptr;
    Meta Infer::meta;

} // namespace lightlda
} // namespace multiverso


int main(int argc, char** argv)
{
    multiverso::lightlda::Config::inference = true;
    multiverso::lightlda::Infer::Run(argc, argv);
    return 0;
}
