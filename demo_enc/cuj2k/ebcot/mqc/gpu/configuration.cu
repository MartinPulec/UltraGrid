/* 
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
 
#include "configuration.h"
#include "common.h"

/** Documented at declaration */
void
mqc_gpu_configuration_load(mqc_gpu_configuration & config, struct mqc_configuration* configuration)
{
    if ( configuration == 0 )
        return;

    std::string cfg = configuration->configuration;
    if ( cfg == "" )
        return;

    int pos = (int)std::string::npos;

    // Load kernel
    config.kernel = 0;

    // Thread work and per count
    pos = cfg.find("N");
    if ( pos == std::string::npos )
        pos = -1;
    std::string tcfg = cfg;
    tcfg.erase(0,pos + 1);
    pos = tcfg.find("/");
    if ( pos != std::string::npos ) {
        config.threadWorkCount = atoi(tcfg.substr(0,pos).c_str());
        tcfg.erase(0,pos + 1);

        pos = tcfg.find("-");
        if ( pos == std::string::npos ) {
            config.threadPerCount = atoi(tcfg.c_str());
        } else {
            config.threadPerCount = atoi(tcfg.substr(0,pos).c_str());
            tcfg.erase(0,pos + 1);
        }
    }

    // CX,D load type
    pos = cfg.find("T");
    if ( pos != std::string::npos ) {
        int byte_count = atoi(cfg.substr(pos + 1,2).c_str());
        switch ( byte_count ) {
            case 1:
                config.cxdLoadType = mqc_gpu_configuration::Byte;
                break;
            case 2:
                config.cxdLoadType = mqc_gpu_configuration::Short;
                break;
            case 4:
                config.cxdLoadType = mqc_gpu_configuration::Integer;
                break;
            case 8:
                config.cxdLoadType = mqc_gpu_configuration::Double;
                break;
        }
    }

    // CX,D load count
    pos = cfg.find("L");
    if ( pos != std::string::npos ) {
        config.cxdLoadCount = atoi(cfg.substr(pos + 1,2).c_str());
    }
    
    // Calculate
    pos = cfg.find("C");
    if ( pos != std::string::npos ) {
        int calculate_count = atoi(cfg.substr(pos + 1,1).c_str());
        switch ( calculate_count ) {
            case 0:
                config.calculate = calculate_none;
                break;
            case 1:
                config.calculate = calculate_once;
                break;
            case 2:
                config.calculate = calculate_twice;
                break;
            case 3:
                config.calculate = calculate_tripple;
                break;
        }
    }
}

/** Documented at declaration */
int
mqc_gpu_configuration_run(mqc_gpu_configuration & config, struct j2k_cblk* d_cblk, int cblk_count, unsigned char * d_cxd, unsigned char * d_byte)
{
    if ( config.kernel == 0 )
        return -1;

    cudaError cuerr = cudaSuccess;

    int count = cblk_count / config.threadWorkCount + 1;

    dim3 dim_grid;
    dim_grid.x = count;
    if ( dim_grid.x > CUDA_MAXIMUM_GRID_SIZE ) {
        dim_grid.x = CUDA_MAXIMUM_GRID_SIZE;
        dim_grid.y = count / CUDA_MAXIMUM_GRID_SIZE + 1;
    }
    dim3 dim_block(config.threadPerCount * config.threadWorkCount,1);

    config.kernel<<<dim_grid,dim_block>>>(d_cblk, cblk_count, d_cxd, d_byte);
    cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        std::cerr << "MQ-Coder kernel encode failed: " << cudaGetErrorString(cuerr) << std::endl;
        return -1;
    }
    return 0;
}

