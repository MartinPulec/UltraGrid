///
/// @file    j2kd_timer.h
/// @author  Martin Jirman (martin.jirman@cesnet.cz)
/// @brief   Helper tool for measuring time of decoder components.
/// 


#ifndef J2KD_TIMER_H
#define J2KD_TIMER_H

#include <sys/time.h>
#include "j2kd_cuda.h"


namespace cuj2kd {
    

/// Gets current time (from some epoch), in milliseconds.
inline double getTimeMs() {
    timeval tv;
    
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000.0 + tv.tv_usec * 0.001;
}


    
/// Measures performance of CUDA kernel executions.
class GPUTimer {
private:
    bool stopped, started;
    cudaEvent_t begin, end;
    const cudaStream_t & stream;
    
public:
    GPUTimer(const cudaStream_t & stream) : stream(stream) {
        stopped = false;
        started = false;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);
    }
    
    ~GPUTimer() {
        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }
    
    void start() {
        if(!started) {
            cudaEventRecord(begin, stream);
            started = true;
        }
    }
    
    void stop() {
        if(started && !stopped) {
            cudaEventRecord(end, stream);
            stopped = true;
        }
    }
    
    double getTime() const {
        float result = -1.0f;
        
        if(started && stopped) {
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&result, begin, end);
        } 
        return result;
    }
}; // end of class GPUTimer




class CPUTimer {
private:
    bool started, stopped;
    double begin, end;
    
public:
    CPUTimer() {
        stopped = false;
        started = false;
    }
    
    void start() {
        if(!started) {
            begin = getTimeMs();
            started = true;
        }
    }
    
    void stop() {
        if(started && !stopped) {
            end = getTimeMs();
            stopped = true;
        }
    }
    
    /// @return time in milliseconds between start and stop calls
    double getTime() const {
        return (started && stopped) ? end - begin : -1.0;
    }
}; // end of class CPUTimer



} // end of namespace cuj2kd

#endif // J2KD_TIMER_H

