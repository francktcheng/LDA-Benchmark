#ifndef MULTIVERSO_MT_QUEUE_H_
#define MULTIVERSO_MT_QUEUE_H_
/*! 
 * \brief Defines a thread safe queue
 */
#include <queue>
#include <mutex>
#include <condition_variable>

namespace multiverso 
{
    /*!
     * \brief A thread safe queue support multithread push and pop concurrently
     *        The queue is based on move semantics.
     */
    template<typename T>
    class MtQueueMove {
    public:
        /*! \brief Constructor */
        MtQueueMove() : exit_(false) {}

        /*! 
         * \brief Push an element into the queue. the function is based on 
         *        move semantics. After you pushed, the item would be an 
         *        uninitialized variable. 
         * \param item item to be pushed
         */
        void Push(T& item);

        /*! 
         * \brief Pop an element from the queue, if the queue is empty, thread
         *        call pop would be blocked
         * \param result the returned result 
         * \return true when pop sucessfully; false when the queue is exited
         */
        bool Pop(T& result);

        /*!
        * \brief Get the front element from the queue, if the queue is empty, 
        *        threat who call front would be blocked. Not move semantics.
        * \param result the returned result
        * \return true when pop sucessfully; false when the queue is exited
        */
        bool Front(T& result);

        /*!
         * \brief Gets the number of elements in the queue
         * \return size of queue
         */
        int Size() const;

        /*! 
         * \brief Whether queue is empty or not
         * \return true if queue is empty; false otherwise
         */
        bool Empty() const;

        /*! \brief Exit queue, awake all threads blocked by the queue */
        void Exit();

    private:
        /*! the underlying container of queue */
        std::queue<T> buffer_;
        mutable std::mutex mutex_;
        std::condition_variable empty_condition_;
        /*! whether the queue is still work */
        bool exit_;

        // No copying allowed
        MtQueueMove(const MtQueueMove&);
        void operator=(const MtQueueMove&);
    };

    template<typename T>
    void MtQueueMove<T>::Push(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        buffer_.push(std::move(item));
        empty_condition_.notify_one();
    }

    template<typename T>
    bool MtQueueMove<T>::Pop(T& result)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        empty_condition_.wait(lock,
            [this]{ return !buffer_.empty() || exit_; });
        if (buffer_.empty())
        {
            return false;
        }
        result = std::move(buffer_.front());
        buffer_.pop();
        return true;
    }

    template<typename T>
    bool MtQueueMove<T>::Front(T& result)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        empty_condition_.wait(lock,
            [this]{ return !buffer_.empty() || exit_; });
        if (buffer_.empty())
        {
            return false;
        }
        result = buffer_.front();
        return true;
    }

    template<typename T>
    int MtQueueMove<T>::Size() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return static_cast<int>(buffer_.size());
    }

    template<typename T>
    bool MtQueueMove<T>::Empty() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return buffer_.empty();
    }

    template<typename T>
    void MtQueueMove<T>::Exit()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        exit_ = true;
        empty_condition_.notify_all();
    }
}

#endif // MULTIVERSO_MT_QUEUE_H_
