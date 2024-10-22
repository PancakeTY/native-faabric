#pragma once

#include <faabric/util/exception.h>
#include <faabric/util/locks.h>
#include <faabric/util/logging.h>

#include <atomic_queue/atomic_queue.h>
#include <condition_variable>
#include <queue>
#include <readerwriterqueue/readerwritercircularbuffer.h>

#define DEFAULT_QUEUE_TIMEOUT_MS 5000
#define DEFAULT_QUEUE_SIZE 1024

namespace faabric::util {
class QueueTimeoutException : public faabric::util::FaabricException
{
  public:
    explicit QueueTimeoutException(std::string message)
      : FaabricException(std::move(message))
    {}
};

template<typename T>
class Queue
{
  public:
    void enqueue(T value)
    {
        UniqueLock lock(mx);

        mq.emplace(std::move(value));

        enqueueNotifier.notify_one();
    }

    void dequeueIfPresent(T* res)
    {
        UniqueLock lock(mx);

        if (!mq.empty()) {
            T value = std::move(mq.front());
            mq.pop();
            emptyNotifier.notify_one();

            *res = value;
        }
    }

    T dequeue(long timeoutMs = DEFAULT_QUEUE_TIMEOUT_MS)
    {
        UniqueLock lock(mx);

        if (timeoutMs <= 0) {
            SPDLOG_ERROR("Invalid queue timeout: {} <= 0", timeoutMs);
            throw std::runtime_error("Invalid queue timeout");
        }

        while (mq.empty()) {
            std::cv_status returnVal = enqueueNotifier.wait_for(
              lock, std::chrono::milliseconds(timeoutMs));

            // Work out if this has returned due to timeout expiring
            if (returnVal == std::cv_status::timeout) {
                throw QueueTimeoutException("Timeout waiting for dequeue");
            }
        }

        T value = std::move(mq.front());
        mq.pop();
        emptyNotifier.notify_one();

        return value;
    }

    T* peek(long timeoutMs = 0)
    {
        UniqueLock lock(mx);
        while (mq.empty()) {
            if (timeoutMs > 0) {
                std::cv_status returnVal = enqueueNotifier.wait_for(
                  lock, std::chrono::milliseconds(timeoutMs));

                if (returnVal == std::cv_status::timeout) {
                    throw QueueTimeoutException("Timeout waiting for dequeue");
                }
            } else {
                enqueueNotifier.wait(lock);
            }
        }

        return &mq.front();
    }

    void waitToDrain(long timeoutMs)
    {
        UniqueLock lock(mx);

        while (!mq.empty()) {
            if (timeoutMs > 0) {
                std::cv_status returnVal = emptyNotifier.wait_for(
                  lock, std::chrono::milliseconds(timeoutMs));

                // Work out if this has returned due to timeout expiring
                if (returnVal == std::cv_status::timeout) {
                    throw QueueTimeoutException("Timeout waiting for empty");
                }
            } else {
                emptyNotifier.wait(lock);
            }
        }
    }

    void drain()
    {
        UniqueLock lock(mx);

        while (!mq.empty()) {
            mq.pop();
        }
    }

    long size()
    {
        UniqueLock lock(mx);
        return mq.size();
    }

    void reset()
    {
        UniqueLock lock(mx);

        std::queue<T> empty;
        std::swap(mq, empty);
    }

  private:
    std::queue<T> mq;
    std::condition_variable enqueueNotifier;
    std::condition_variable emptyNotifier;
    std::mutex mx;
};

// Wrapper around moodycamel's blocking fixed capacity single producer single
// consumer queue
// https://github.com/cameron314/readerwriterqueue
template<typename T>
class FixedCapacityQueue
{
  public:
    FixedCapacityQueue(int capacity)
      : mq(capacity){};

    FixedCapacityQueue()
      : mq(DEFAULT_QUEUE_SIZE){};

    void enqueue(T value, long timeoutMs = DEFAULT_QUEUE_TIMEOUT_MS)
    {
        if (timeoutMs <= 0) {
            SPDLOG_ERROR("Invalid queue timeout: {} <= 0", timeoutMs);
            throw std::runtime_error("Invalid queue timeout");
        }

        bool success =
          mq.wait_enqueue_timed(std::move(value), timeoutMs * 1000);
        if (!success) {
            throw QueueTimeoutException("Timeout waiting for enqueue");
        }
    }

    void dequeueIfPresent(T* res) { mq.try_dequeue(*res); }

    T dequeue(long timeoutMs = DEFAULT_QUEUE_TIMEOUT_MS)
    {
        if (timeoutMs <= 0) {
            SPDLOG_ERROR("Invalid queue timeout: {} <= 0", timeoutMs);
            throw std::runtime_error("Invalid queue timeout");
        }

        T value;
        bool success = mq.wait_dequeue_timed(value, timeoutMs * 1000);
        if (!success) {
            throw QueueTimeoutException("Timeout waiting for dequeue");
        }

        return value;
    }

    T* peek(long timeoutMs = DEFAULT_QUEUE_TIMEOUT_MS)
    {
        throw std::runtime_error("Peek not implemented");
    }

    void drain(long timeoutMs = DEFAULT_QUEUE_TIMEOUT_MS)
    {
        T value;
        bool success;
        while (size() > 0) {
            success = mq.wait_dequeue_timed(value, timeoutMs * 1000);
            if (!success) {
                throw QueueTimeoutException("Timeout waiting to drain");
            }
        }
    }

    long size() { return mq.size_approx(); }

    void reset()
    {
        moodycamel::BlockingReaderWriterCircularBuffer<T> empty(
          mq.max_capacity());
        std::swap(mq, empty);
    }

  private:
    moodycamel::BlockingReaderWriterCircularBuffer<T> mq;
};

template<typename T>
class SpinLockQueue
{
  public:
    void enqueue(T& value) { mq.push(value); }

    T dequeue() { return mq.pop(); }

    long size()
    {
        throw std::runtime_error("Size for fast queue unimplemented!");
    }

    void drain()
    {
        while (mq.pop()) {
            ;
        }
    }

    void reset() { ; }

  private:
    atomic_queue::AtomicQueue2<T, 1024, true, true, false, true> mq;
};

class TokenPool
{
  public:
    explicit TokenPool(int nTokens);

    int getToken();

    void releaseToken(int token);

    void reset();

    int size();

    int taken();

    int free();

  private:
    int _size;
    Queue<int> queue;
};

template<typename T>
class ThreadSafeQueue
{
  public:
    ThreadSafeQueue() = default;

    // Disable copy constructor and assignment operator
    ThreadSafeQueue(const ThreadSafeQueue&) = delete;
    ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

    // Enqueue an item
    void enqueue(T item)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(std::move(item));
        m_cond_var.notify_one();
    }

    // Get the item at the front of the queue without removing it. Blocks if the queue is empty.
    T front()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond_var.wait(lock, [this]() { return !m_queue.empty(); });
        return m_queue.front();
    }

    // Dequeue an item, blocks if the queue is empty
    T dequeue()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_cond_var.wait(lock, [this]() { return !m_queue.empty(); });
        T item = std::move(m_queue.front());
        m_queue.pop();
        return item;
    }

    // Try to dequeue an item, returns false if the queue is empty
    bool try_dequeue(T& item)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty()) {
            return false;
        }
        item = std::move(m_queue.front());
        m_queue.pop();
        return true;
    }

    // Check if the queue is empty
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }

    // Get the size of the queue
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

  private:
    std::queue<T> m_queue;
    mutable std::mutex m_mutex;
    std::condition_variable m_cond_var;
};
}
