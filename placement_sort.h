/*
 * Copyright 2018-2019 Alexandr Kobotov
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * PLACEMENT SORT:
 * A fast sorting algorith at average O(n) and worst O(n*log(n)) operations.
 */

#pragma once

#include <memory>
#include <vector>
#include <array>
#include <limits>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace placement_sort {

/*
 * Forward declarations (skip that)
 */

namespace internals {

template <bool use_buffer, typename T, typename TValueAccessor>
class ElementAccessor;

template<typename TElementAccessor>
void placement_sort(TElementAccessor& array);

} // namespace internals


/*
 * Base interface functions
 */

template <typename T, typename TValueAccessor>
inline void sort(T* first, size_t count, const TValueAccessor& valueAccessor) {
    placement_sort::internals::ElementAccessor<true, T, TValueAccessor> array(first, count, valueAccessor);
    placement_sort::internals::placement_sort(array);
}


// Version for use with iterators.
template <class RandomIt, typename TValueAccessor>
inline void sort(RandomIt first, RandomIt last, const TValueAccessor& valueAccessor) {
    placement_sort::internals::ElementAccessor<true, typename RandomIt::value_type, TValueAccessor> array(&first[0], last - first, valueAccessor);
    placement_sort::internals::placement_sort(array);
}


// TValueAccessor for arrays of fundamental types (int, float, etc..).
template <class T>
struct Ascending {
    T operator() (const T* val) const { return *val;}
};


// TValueAccessor for reverse sorting.
template <class T, typename T_is_integral = void>
struct Descending {
    T operator() (const T* val) const { return -*val;}
};


// TValueAccessor for reverse sortin of unsigned types.
template <class T>
struct Descending<T, typename std::enable_if<std::is_unsigned<T>::value, void>::type> {
    T operator() (const T* val) const { return std::numeric_limits<T>::max() - *val;}
};


// Ascendong sort of raw arrays of fundamental types.
template <typename T>
inline void sort(T* first, size_t count) {
    placement_sort::sort(first, count, Ascending<T>());
}


// Asceding sort of RandomIt[first,last).
template <class RandomIt>
inline void sort(RandomIt first, RandomIt last) {
    placement_sort::sort(first, last, Ascending<typename RandomIt::value_type>());
}


/* --- Aliases --- */


template <class RandomIt>
inline void stable_sort(RandomIt first, RandomIt last) {
    placement_sort::sort(first, last);
}


// Ascending stable sort by a custom field/method.
template <class RandomIt, typename TValueAccessor>
inline void stable_sort(RandomIt first, RandomIt last, const TValueAccessor& valueAccessor) {
    placement_sort::sort(first, last, valueAccessor);
}


// Descending sort of raw arrays of fundamental types.
template <typename T>
inline void reverse_sort(T* first, size_t count) {
    placement_sort::sort(first, count, Descending<T>());
}


// Descending sort of RandomIt[first, last).
template <class RandomIt>
inline void reverse_sort(RandomIt first, RandomIt last) {
    placement_sort::sort(first, last, Descending<typename RandomIt::value_type>());
}



/*****************************************************
 *
 * Internal classes and functions.
 * Cross this line to learn how the algorithm works.
 *
 *****************************************************/

namespace internals {

#if __cpp_if_constexpr
#define _PLACEMENT_SORT_CONSTEXPR constexpr
#else
#define _PLACEMENT_SORT_CONSTEXPR
#endif

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(name) 0
#endif

#if __has_cpp_attribute(nodiscard)
#define _PLACEMENT_SORT_NODISCARD [[nodiscard]]
#elif __has_cpp_attribute(gnu::warn_unused_result)
#define _PLACEMENT_SORT_NODISCARD [[gnu::warn_unused_result]]
#else
#define _PLACEMENT_SORT_NODISCARD
#endif


template<typename TElementAccessor>
void placement_sort(TElementAccessor& array);

template<typename T>
class SharedUninitializedBuffer {
    /*
     * RAII holder for _uninitialized_ memory buffer.
     * For a use as temp storage with std::move
     */
    public:
        SharedUninitializedBuffer() : ptr(nullptr), count_refs(nullptr) {};
        SharedUninitializedBuffer(SharedUninitializedBuffer& other) : ptr(other.ptr), count_refs(other.count_refs) {
            if (ptr != nullptr)
                ++*count_refs;
        }

        SharedUninitializedBuffer(size_t size) : ptr(nullptr), count_refs(nullptr) {
            if (size) {
                ptr = (T*) malloc(sizeof(T) * size + sizeof(size_t));
                if (ptr == nullptr) {
                    throw std::bad_alloc();
                }
                count_refs = (size_t*) (ptr + size);
                *count_refs = 1;
            }
        }

        inline T& operator[] (size_t i) {
            return ptr[i];
        }

        inline T& operator[] (size_t i) const {
            return ptr[i];
        }

        ~SharedUninitializedBuffer() {
            if (ptr != nullptr) {
                --*count_refs;
                if (*count_refs == 0) {
                    free(ptr);
                    ptr = nullptr;
                    count_refs = nullptr;
                }
            }
        }

    private:
        T* ptr;
        size_t* count_refs;
};


template <bool use_buffer, typename T, typename TValueAccessor>
class ElementAccessor {
    /*
     * Holder for an array, a function to access values to use for soring (esp usefull
     * for structs), and a temporal memory buffer (if needed).
     */
    public:
        using value_type = typename std::result_of<TValueAccessor(const T* val)>::type;

        ElementAccessor(T* array, size_t count, const TValueAccessor& value_accessor) :
            array(array), count(count),
            value_accessor(value_accessor), buffer(use_buffer ? count : 0) {
        }

        /* Creates subinterval, which points to the same array, and reuses already allocatad internal buffer */
        ElementAccessor(ElementAccessor& other, size_t begin_sub_interval, size_t length_sub_interval) :
            array(&other.array[begin_sub_interval]), count(length_sub_interval),
            value_accessor(other.value_accessor), buffer(other.buffer) {
        }

        inline value_type get_value(size_t i) const {
            return value_accessor(&array[i]);
        }

        inline size_t get_count() const {
            return count;
        }

        inline void
        swap(size_t i, size_t j) {
            if (i != j)
                std::swap(array[i], array[j]);
        }

        template<bool has_buffer = use_buffer>
        inline typename std::enable_if<has_buffer, void>::type
        move_to_buffer(size_t i) {
            buffer[i] = std::move(array[i]);
        }

        template<bool has_buffer = use_buffer>
        inline typename std::enable_if<has_buffer, void>::type
        move_from_buffer(size_t position_buffer, size_t position_array) {
            array[position_array] = std::move(buffer[position_buffer]);
        }

        template<bool has_buffer = use_buffer>
        inline typename std::enable_if<has_buffer, value_type>::type
        get_buf_value(size_t i) const {
             return value_accessor(&buffer[i]);
        }

        constexpr static inline bool uses_buffer() {
            return use_buffer;
        }

    private:
        T* array;
        const size_t count;
        const TValueAccessor& value_accessor;
        SharedUninitializedBuffer<T> buffer;
};


template<typename T>
inline bool isfinite(T& t) {
    return true;
}


template<>
inline bool isfinite<double>(double& t) {
    return std::isfinite(t);
}

template<>
inline bool isfinite<float>(float& t) {
    return std::isfinite(t);
}



template <typename T, typename TElementAccessor>
class Statistics {
    /* Finds finite min and max values in the array
     * and meanwhile check if it is already sorted
     */
    public:
        using value_type = T;
        Statistics(const TElementAccessor& array) {
            T prev_value = array.get_value(0);
            const size_t size = array.get_count();
            size_t first_finite_i = 0;
            sorted = true;

            if _PLACEMENT_SORT_CONSTEXPR (std::numeric_limits<T>::has_infinity) {
                while(!isfinite(prev_value) && first_finite_i < size) {
                    ++first_finite_i;
                    const T value = array.get_value(first_finite_i);
                    if (value < prev_value)
                        sorted = false;
                    prev_value = value;
                }
            }

            min = prev_value;
            max = prev_value;

            for(size_t i = first_finite_i + 1; i < size; i++) {
                const T value = array.get_value(i);
                if (value < prev_value)
                    sorted = false;
                prev_value = value;
                if _PLACEMENT_SORT_CONSTEXPR (std::numeric_limits<T>::has_infinity)
                    if (!isfinite(value))
                        continue;
                if (value < min)
                    min = value;
                if (max < value)
                    max = value;
            }
        }

        inline const T& get_min() const {
            return min;
        }

        inline const T& get_max() const {
            return max;
        }

        bool is_sorted() const {
            return sorted;
        }

    private:
        T min;
        T max;
        bool sorted;
};


template <typename T, typename TStatistics, typename T_is_integral = void>
class PlaceCalculator {
    /* Compute destination place for an element in sorted array as
     *    place = (element - min) * size / (max - min).
     * This implementation is for integer types
     */
    public:
        PlaceCalculator(const TStatistics& statistics, size_t size) : min(statistics.get_min()) {
            T max = statistics.get_max();
            invariant = ((((long long int)size) << 32) - 1) / ((long long int)max - min);
        }

        inline size_t get_place(const T& element) const {
             /* Division below is replaced with multiplication and bit shift for performance reasons. */
            return (size_t)((((long long int)element - min) * invariant) >> 32);
        }

    private:
        T min;
        unsigned long long int invariant;
};


template <typename T, typename TStatistics>
class PlaceCalculator<T, TStatistics, typename std::enable_if<std::is_floating_point<T>::value, void>::type> {
    /* Compute destination place for an element in sorted array as
     *    place = (element - min) * size / (max - min).
     * This implementation is for real types (float, double, ...).
     */
   public:
        PlaceCalculator(const TStatistics& statistics, size_t size) : min(statistics.get_min()), last_index(size - 1) {
            T max = statistics.get_max();
            invariant = ((long double)size - 1.) / (max - min);
            if (!isfinite(invariant) || invariant == 0.) {
                split = true;
                split_value = (0.5 * max + 0.5 * min);
                if (!isfinite(split_value))
                    split_value = max;
            }
        }

        inline size_t get_place(const T& element) const {
            /* Division below is replaced with multiplication for performance reasons */
            if _PLACEMENT_SORT_CONSTEXPR (std::numeric_limits<T>::has_infinity) {
                if (element == std::numeric_limits<T>::infinity())
                    return last_index;
                if (element == -std::numeric_limits<T>::infinity())
                    return 0;
            }
            if (split) {
                return (element < split_value) ? 0 : last_index;
            }
            return (size_t) ((element - min) * invariant);
        }

    private:
        T min;
        long double invariant;
        size_t last_index;
        bool split = false;
        T split_value;
};

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
template<typename T>
static inline void prefetch(T p) {
    _mm_prefetch((char*)p, _MM_HINT_NTA);
}
#else
template<typename T>
static inline void prefetch(T p) {
}
#endif

template <bool fill_buffer, typename counters_t, typename TElementAccessor,typename TPlaceCalculator>
static inline void count_values_at_each_place(TElementAccessor& array, const TPlaceCalculator& placer,
                                              counters_t& counters, bool& has_collisions) {
    const size_t size = array.get_count();
    constexpr size_t prefetch_step = 128;
    has_collisions = false;
    for (size_t i = 0; i < size; i++) {
        if (sizeof(typename counters_t::value_type) > 2) if (i + prefetch_step < size)
            prefetch(counters.data() + placer.get_place(array.get_value(i + prefetch_step)));
        size_t place = placer.get_place(array.get_value(i));
        ++counters[place];
        has_collisions |= counters[place] > 1;

        /* Moving to a buffer is included here for performance reasons (reuse hot data in cache) */
        if _PLACEMENT_SORT_CONSTEXPR (fill_buffer)
            array.move_to_buffer(i);;
    }
}


template <typename counters_t>
static inline void compute_memory_distribution(counters_t& counters) {
        typename counters_t::value_type position = 0;
        for (auto& counter: counters) {
            auto count = counter;
            counter = position;
            position += count;
        }
}


template <typename TElementAccessor, typename TPlaceCalculator, typename counters_t>
static inline void move_elements_out_of_place(TElementAccessor& array, const TPlaceCalculator& placer, counters_t& counters) {
    const size_t size = array.get_count();
    for (size_t i = 0; i < size; i++) {
        if (sizeof(typename counters_t::value_type) > 2) if (i + 128 < size)
            prefetch(&counters[placer.get_place(array.get_buf_value(i + 128))]);
        size_t target_place = placer.get_place(array.get_buf_value(i));
        size_t actual_place = counters[target_place]++;
        array.move_from_buffer(i, actual_place);
    }
}


template <typename TElementAccessor, typename TPlaceCalculator, typename counters_t>
static inline void move_elements_in_place(TElementAccessor& array, const TPlaceCalculator& placer, counters_t& counters) {
    const size_t size = array.get_count();
    constexpr typename counters_t::value_type topBit = (typename counters_t::value_type)1 << (sizeof(typename counters_t::value_type)*8 - 1);
    constexpr size_t block_size_high = 32;
    constexpr size_t block_size_low = 4;
    const size_t block_size = (size > 512*1024) ? block_size_high : block_size_low;

    /* This algorithm moves elements to their places and sorts out collisions with no extra memory except already available counters.
     * It uses highest bit in counters to mark elements which are already moved to their destination place.
     * This way is fastest though looks ugly. It saves memory traffic. */

    for (size_t sorted = 0; sorted < size; ) {
        size_t places[block_size_high];
        const size_t block_end = sorted + std::min(block_size, size - sorted);
        for (size_t i = sorted; i < block_end; ++i) { // prefetch counters
            if (!(topBit & counters[i])) {
                const size_t place = placer.get_place(array.get_value(i));
                places[i - sorted] = place;
                prefetch(&counters[place]);
            }
        }
        for (size_t i = sorted; i < block_end; ++i) { // move
            if (!(topBit & counters[i])) {
                const size_t target_place = places[i - sorted];
                const size_t actual_place = ~topBit & counters[target_place];
                array.swap(i, actual_place);
                counters[actual_place] |= topBit;
                counters[target_place]++;
            }
        }
        while (sorted < size && topBit & counters[sorted]) {
            counters[sorted] &= ~topBit;
            sorted++;
        }
    }
}


template <typename TElementAccessor, typename TPlaceCalculator, typename counters_t>
static inline void move_elements(TElementAccessor& array, const TPlaceCalculator& placer, counters_t& counters) {
    /*
     * Move elements according computed memory distribution for colliding elements.
     */
    if _PLACEMENT_SORT_CONSTEXPR (TElementAccessor::uses_buffer()) {
        move_elements_out_of_place(array, placer, counters);
    } else {
        move_elements_in_place(array, placer, counters);
    }
}


template <typename TElementAccessor, typename TPlaceCalculator>
static inline void move_elements(TElementAccessor& array, const TPlaceCalculator& placer) {
    /* This version to be called only when there are no collisions */
    const size_t size = array.get_count();
    for (size_t i = 0; i < size; i++) {
        size_t place = placer.get_place(array.get_value(i));
        while (place != i) {
            array.swap(i, place);
            place = placer.get_place(array.get_value(i));
        }
    }
}


template<typename TElementAccessor>
static inline void selection_sort(TElementAccessor& array);

template<typename  TElementAccessor>
void merge_sort(TElementAccessor& array);

template<typename  TElementAccessor>
void quick_sort(TElementAccessor& array);


template <typename TElementAccessor, typename counters_t>
static inline void sort_collisions(TElementAccessor& array, counters_t& counters) {
    const size_t size = array.get_count();
    typename counters_t::value_type position = 0;
    for (size_t i = 0; i < size; i++) {
        typename counters_t::value_type count = counters[i] - position;
        if (count > 1) {
            TElementAccessor sub_interval(array, position, count);
            bool placer_is_not_optimal = count > size / 2;
            if (placer_is_not_optimal) {
                if _PLACEMENT_SORT_CONSTEXPR(TElementAccessor::uses_buffer()) {
                    merge_sort(sub_interval);
                } else {
                    quick_sort(sub_interval);
                }
            } else {
                placement_sort(sub_interval);
            }
        }
        position += count;
    }
}


/*
 * Selection sort has minimal swaps, is stable and is good for modern vectorization.
 */
template<typename TElementAccessor>
static inline void selection_sort(TElementAccessor& array) {
    const size_t size = array.get_count();
    bool is_sorted = false;
    for (size_t i = 0; i < size - 1 && !is_sorted; i++) {
        auto min_val = array.get_value(i);
        auto prev_val = min_val;
        size_t min_idx = i;
        is_sorted = true;
        for (size_t j = i + 1; j < size; j++) {
            auto cur_val = array.get_value(j);
            if (cur_val < min_val) {
                min_val = cur_val;
                min_idx = j;
            }
            if (cur_val < prev_val) {
                is_sorted = false;
            }
            prev_val = cur_val;

        }
        if (min_idx != i)
            array.swap(i, min_idx);
    }
}


template<typename TElementAccessor>
static inline void three_sort(TElementAccessor& array, size_t index_0, size_t index_1, size_t index_2) {
    if (array.get_value(index_1) < array.get_value(index_0))
        array.swap(index_1, index_0);
    if (array.get_value(index_2) < array.get_value(index_0))
        array.swap(index_2, index_0);
    if (array.get_value(index_2) < array.get_value(index_1))
        array.swap(index_2, index_1);
}

template<typename TElementAccessor>
static inline bool small_size_sort(TElementAccessor& array) {
    const size_t size = array.get_count();
    switch(size) {
        case 0:
        case 1:
            return true;
        case 2:
            if (array.get_value(1) < array.get_value(0))
                array.swap(1, 0);
            return true;
        case 3:
            three_sort(array, 0, 1, 2);
            return true;
        default:
            if (size <= 8) {
                selection_sort(array);
                return true;
            }
    }

    return false; // Means array was not sorted
}


/*
 * Merge sort guards upper O(N*log(N)) operations limit,
 * and it is stable (preserves equal elements order).
 */
template<typename  TElementAccessor>
void merge_sort(TElementAccessor& array) {
    if (small_size_sort(array))
        return;

    const size_t size = array.get_count();
    const size_t half = size / 2;
    {
        TElementAccessor sub_interval_left(array, 0, half);
        placement_sort(sub_interval_left);
    }
    {
        TElementAccessor sub_interval_right(array, half, size - half);
        placement_sort(sub_interval_right);
    }

    for(size_t i = 0; i < size; ++i) {
        array.move_to_buffer(i);
    }

    for(size_t left = 0, right = half, i = 0; i < size; ++i) {
        if (left == half || (right < size && array.get_buf_value(right) < array.get_buf_value(left))) {
            array.move_from_buffer(right, i);
            ++right;
        } else {
            array.move_from_buffer(left, i);
            ++left;
        }
    }
}


template<typename TElementAccessor>
void quick_sort(TElementAccessor& array) {
    if (small_size_sort(array))
        return;

    const size_t size = array.get_count();
    const size_t mid = size / 2;
    const size_t top = size - 1;

    three_sort(array, 0, top, mid);

    typename TElementAccessor::value_type pivot = array.get_value(top);
    size_t i = 0, j = top - 1;
    while (i < j) {
        while (i <= j && array.get_value(i) <= pivot) i++;
        while (i <= j && array.get_value(j) >= pivot) j--;
        if (i < j)
            array.swap(i++, j--);
    }
    array.swap(i, top);

    if (i > 1) {
        TElementAccessor left(array, 0, i);
        placement_sort(left);
    }
    if (size - i > 2) {
        TElementAccessor right(array, i + 1 , size - i - 1);
        placement_sort(right);
    }
}

/*
 * Placement sort main body
 * @param array       Proxy object to accesss elements to sort and their values to define order
 * @param statistics  Object to hold min and max
 * @param counters    Buffer for counters all set to zero
 */
template <typename counters_t, typename TElementAccessor, typename TStatistics>
static inline void placement_sort_body(TElementAccessor& array, const TStatistics& statistics, counters_t& counters) {
    bool has_collisions;
    const size_t size = array.get_count();

    const PlaceCalculator<typename TElementAccessor::value_type, TStatistics> placer(statistics, size);

    count_values_at_each_place<TElementAccessor::uses_buffer()>(array, placer, counters, has_collisions);

    if (has_collisions) {
        compute_memory_distribution(counters);
        move_elements(array, placer, counters);
        sort_collisions(array, counters);
    } else {
        move_elements(array, placer);
    }
}


/*
 * Internal entry point. It's a dispatcher to a fastest algorithm.
 * @param array Proxy object to accesss elements to sort and their values to define order
 */
template<typename TElementAccessor>
void placement_sort(TElementAccessor& array) {
    if (small_size_sort(array))
        return;

    Statistics<typename TElementAccessor::value_type, TElementAccessor> statistics(array);

    if (statistics.is_sorted())
        return;

    /* Select an instance to minimize collision counters memory traffic */
    static constexpr size_t MAX_STACK_BUF_SIZE = 1; // 2048; std::array is slow
    const size_t size = array.get_count();
    if (size < MAX_STACK_BUF_SIZE) {
        std::array<unsigned short, MAX_STACK_BUF_SIZE> counters_workspace;
        std::fill(counters_workspace.begin(), counters_workspace.begin() + size, 0);
        placement_sort_body(array, statistics, counters_workspace);
    } else if (size < std::numeric_limits<unsigned short>::max() >> 1) {
        std::vector<unsigned short> counters_workspace(size);
        placement_sort_body(array, statistics, counters_workspace);
    } else if (size < std::numeric_limits<unsigned int>::max() >> 1) {
        std::vector<unsigned int> counters_workspace(size);
        placement_sort_body(array, statistics, counters_workspace);
    } else {
        std::vector<unsigned long long int> counters_workspace(size);
        placement_sort_body(array, statistics, counters_workspace);
    }
}

/* TODO:
 * Fix MSVS low performance
 * port tests
 * topBit hider functors
 * fix (36, initFexpGrowth<float> non buff case
 * sort(vector<T>) using vector.swap to save moves twice
 * nan support
 * use only < as comparator
 */
} // namespace internals

} // namespace placement_sort
