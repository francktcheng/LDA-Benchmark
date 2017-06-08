#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif
#include <ctime>
#include "stop_watch.h"

namespace multiverso
{
    StopWatch::StopWatch()
    {
        tick_per_sec_ = GetTickPerSec();
        Restart();
    }

    StopWatch::~StopWatch() {}

    uint64_t StopWatch::timenow(void)
    {
      struct timeval tv;
      gettimeofday(&tv, NULL);
      return ((uint64_t)(tv.tv_sec) * 1000000 + tv.tv_usec);
    }

    void StopWatch::Restart()
    {
        elapsed_tick_ = 0;
        start_tick_ = -1;
        Start();
    }

    void StopWatch::Start()
    {
        if (start_tick_ < 0)
        {
            start_tick_ = GetCurrentTick();
        }


        //clock_gettime(CLOCK_MONOTONIC, &ts1);
        ts1 = timenow();
    }

    void StopWatch::Stop()
    {
        if (start_tick_ >= 0)
        {
            elapsed_tick_ += GetCurrentTick() - start_tick_;
            start_tick_ = -1;
        }

        //clock_gettime(CLOCK_MONOTONIC, &ts2);
    }

    bool StopWatch::IsRunning()
    {
        return start_tick_ >= 0;
    }

    double StopWatch::ElapsedSeconds()
    {
//        int64_t addition = IsRunning() ? (GetCurrentTick() - start_tick_) : 0;
//        return static_cast<double>(elapsed_tick_ + addition) / tick_per_sec_;
  
        //clock_gettime(CLOCK_MONOTONIC, &ts2);
        //return ts2.tv_sec - ts1.tv_sec;
        ts2 = timenow();
        return (ts2-ts1)/1000000.0;

    }

#if defined(_WIN32) || defined(_WIN64)
    int64_t StopWatch::GetTickPerSec()
    {
        LARGE_INTEGER tmp;
        QueryPerformanceFrequency(&tmp);
        return tmp.QuadPart;
    }

    int64_t StopWatch::GetCurrentTick()
    {
        LARGE_INTEGER tmp;
        QueryPerformanceCounter(&tmp);
        return tmp.QuadPart;
    }

#else
    int64_t StopWatch::GetTickPerSec()
    {
        return CLOCKS_PER_SEC;
    }

    int64_t StopWatch::GetCurrentTick()
    {
        return clock();
    }

#endif
}
