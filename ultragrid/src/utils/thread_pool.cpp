/*
 * FILE:    utils/ring_buffer.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "utils/thread_pool.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <queue>

static void *worker(void *arg);

using namespace std;

struct thread_pool {
        queue<void *> incoming;
        queue<void *> outgoing;
        pthread_t        *threads;
        bool             *occupied;
        bool             *worker_waiting;
        void            **jobs;
        pthread_cond_t   *cv;
        job_processing_t processing;
        pthread_mutex_t lock;
        volatile bool   new_job;
        volatile bool   boss_waiting;
        pthread_cond_t  boss_cv;

        int thread_count;

        bool should_exit;

        thread_pool(int count, job_processing_t job_processing) {
                processing = job_processing;
                should_exit = false;

                threads = (pthread_t *) malloc(count * sizeof(pthread_t));
                occupied = (bool *) malloc(count * sizeof(bool));
                worker_waiting = (bool *) malloc(count * sizeof(bool));
                jobs = (void **) malloc(count * sizeof(void *));
                for (int i = 0; i < count; ++i) {
                        occupied[i] = false;
                        worker_waiting[i] = false;
                }
                cv = (pthread_cond_t *) malloc(count * sizeof(pthread_cond_t));
                for (int i = 0; i < count; ++i) {
                         pthread_cond_init(&cv[i], NULL);
                }

                boss_waiting = false;
                pthread_cond_init(&boss_cv, NULL);

                thread_count = 0;
                pthread_mutex_init(&lock, NULL);
                for (int i = 0; i < count; ++i) {
                        pthread_mutex_lock(&lock);
                        pthread_create(&threads[i], NULL, worker, this);
                }
        }

        ~thread_pool() {
                delete [] threads;
                delete [] occupied;
                delete [] worker_waiting;
                delete [] jobs;
        }

        void enqueue(void *job) {
                pthread_mutex_lock(&lock);
                for(int i = 0; i < thread_count; i++) {
                        if(occupied[i] == false) {
                                occupied[i] = true;
                                jobs[i] = job;
                                if(worker_waiting[i])
                                        pthread_cond_signal(&cv[i]);
                                pthread_mutex_unlock(&lock);
                                return;
                        }
                }

                /* or we just put it int queue */
                incoming.push(job);
                pthread_mutex_unlock(&lock);
        }

        void * pop() {
                void *res;

                pthread_mutex_lock(&lock);
                if(outgoing.size() == 0) {
                        int wait_count = 0;
                        for (int i = 0; i < thread_count; ++i) {
                                if(occupied[i])
                                        wait_count++;
                        }
                        assert(wait_count > 0);

                        new_job = false;
                        while(!new_job) {
                                boss_waiting = true;
                                pthread_cond_wait(&boss_cv, &lock);
                                boss_waiting = false;
                        }
                }
                res = outgoing.front();
                outgoing.pop();
                pthread_mutex_unlock(&lock);

                return res;
        }

        void update() {
                int i = 0;
                pthread_mutex_lock(&lock);
                while(incoming.size() > 0 && i < thread_count) {
                        if(!occupied[i]) {
                                occupied[i] = true;
                                jobs[i] = incoming.front();
                                incoming.pop();
                                if(worker_waiting[i])
                                        pthread_cond_signal(&cv[i]);
                        }
                        i++;
                }
                pthread_mutex_unlock(&lock);
        }

        void flush() {
                pthread_mutex_lock(&lock);
                while(incoming.size() > 0) {
                        incoming.pop();
                }

                int wait_count = 0;
                for (int i = 0; i < thread_count; ++i) {
                        if(occupied[i])
                                wait_count++;
                }
                pthread_mutex_unlock(&lock);

                for (int i = 0; i < wait_count; ++i) {
                        pop();
                }
        }

        int get_overall_count()
        {
                int overall = 0;
                pthread_mutex_lock(&lock);
                overall = incoming.size() + outgoing.size();
                int wait_count = 0;
                for (int i = 0; i < thread_count; ++i) {
                        if(occupied[i])
                                wait_count++;
                }
                overall += wait_count;
                pthread_mutex_unlock(&lock);

                return overall;
        }
};

static void *worker(void *arg) {
        thread_pool *pool = (thread_pool *) arg;
        int my_id = pool->thread_count++;

        pthread_mutex_unlock(&pool->lock);

        while(!pool->should_exit) {
                pthread_mutex_lock(&pool->lock);
                while(!pool->occupied[my_id]) {
                        pool->worker_waiting[my_id] = true;
                        pthread_cond_wait(&pool->cv[my_id], &pool->lock);
                        pool->worker_waiting[my_id] = false;
                }
                pthread_mutex_unlock(&pool->lock);

                void *res = pool->processing(pool->jobs[my_id]);

                pthread_mutex_lock(&pool->lock);
                pool->occupied[my_id] = false;
                pool->outgoing.push(res);
                pool->new_job = true;
                if(pool->boss_waiting) {
                        pthread_cond_signal(&pool->boss_cv);
                }
                pthread_mutex_unlock(&pool->lock);

                pool->update();
        }
}


struct thread_pool *thread_pool_init(int count, job_processing_t job_processing)
{
        struct thread_pool *pool;

        pool = new thread_pool(count, job_processing);

        return pool;
}

void thread_pool_enqueue(struct thread_pool *pool, void *job)
{
        pool->enqueue(job);
}

void * thread_pool_pop(struct thread_pool *pool)
{
        return pool->pop();
}

void thread_pool_destroy(struct thread_pool * pool)
{
        delete pool;
}

void thread_pool_flush(struct thread_pool *pool)
{
        pool->flush();
}

int thread_pool_get_overall_count(struct thread_pool *pool)
{
        return pool->get_overall_count();
}

