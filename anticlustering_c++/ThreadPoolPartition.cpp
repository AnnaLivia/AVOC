#include <thread>
#include <condition_variable>
#include <vector>
#include "ThreadPoolPartition.h"
#include "config_params.h"
#include "ac_heuristic.h"

// Constructor initializes the thread pool with a given number of threads
ThreadPoolPartition::ThreadPoolPartition(SharedDataPartition* shared_data, int n_thread)
    : shared_data(shared_data), done(false)
{
    const int numberOfThreads = n_thread;
    threads.reserve(numberOfThreads);
    for (int i = 0; i < numberOfThreads; ++i) {
        threads.emplace_back(&ThreadPoolPartition::doWork, this, i);
    }
}

// The destructor joins all the threads so the program can exit gracefully
void ThreadPoolPartition::quitPool() {
    {
        std::lock_guard<std::mutex> l(shared_data->queueMutex);
        done = true;
    }
    shared_data->queueConditionVariable.notify_all();
    for (auto& th : threads) th.join();
}

// Function to add a job to the thread pool's queue
void ThreadPoolPartition::addJob(PartitionJob* job) {
    {
        std::lock_guard<std::mutex> l(shared_data->queueMutex);
        shared_data->queue.push_back(job);
    }
    shared_data->queueConditionVariable.notify_one();
}

// Function used by the threads to grab work from the queue
void ThreadPoolPartition::doWork(int id) {
    // Reuse big buffers per thread to avoid repeated heap allocs
    thread_local arma::mat sol;
    thread_local arma::mat cls;

    for (;;) {
        PartitionJob* job = nullptr;

        // ---- Grab one job or exit ----
        {
            std::unique_lock<std::mutex> l(shared_data->queueMutex);
            shared_data->queueConditionVariable.wait(
                l, [&] { return done || !shared_data->queue.empty(); }
            );
            // Exit only when we're done AND nothing left to do
            if (done && shared_data->queue.empty()) break;

            job = shared_data->queue.front();
            shared_data->queue.pop_front();
            shared_data->threadStates[id] = true;
        }

        // ---- Process job (no locks) ----
        const int np = static_cast<int>(job->part_data.n_rows);
        double ub_mssc = 0.0;
        UserConstraints constraints;
        const double lb_mssc = sdp_branch_and_bound(
            k, job->part_data, ub_mssc, constraints, sol,
            job->kmeans_ub, shared_data->print
        );

        // ---- Build labels + cls fast (vectorized) ----
        arma::uvec labels = arma::index_max(sol, 1);

        // cls = [data | labels]
        cls.set_size(np, d + 1);
        cls.cols(0, d - 1) = job->part_data;
        cls.col(d) = arma::conv_to<arma::colvec>::from(labels);

        // ---- Publish results (very short lock) ----
        {
            std::lock_guard<std::mutex> l(shared_data->queueMutex);
            shared_data->lb_part[job->part_id] = lb_mssc;
            shared_data->ub_part[job->part_id] = ub_mssc;
            shared_data->sol_part[job->part_id] = cls; // leave move if your container supports it
            shared_data->threadStates[id] = false;
        }

        shared_data->mainConditionVariable.notify_one();
        delete job;
    }
}
