Placement sort
==============

A fast O(n) stable sorting algorithm.


Interface
---------

```cpp
template <typename T>
void placement_sort::sort(T* first, size_t count);

template <class RandomIt>
void placement_sort::sort(RandomIt first, RandomIt last); 

template <typename T>
void placement_sort::reverse_sort(T* first, size_t count);

template <class RandomIt>
void placement_sort::reverse_sort(RandomIt first, RandomIt last); 
```
Above is similar to std::sort interface.
Requires `operator[](size_t)` to return a fundamental type (int, float, etc..), and will sort basing on the value.
Also requires T or dereferenced RandomIt to support std::move.


Below are interfaces with custom accessor to value to sort on:
```cpp
template <typename T, typename TValueAccessor>
void placement_sort::sort(T* first, size_t count, const TValueAccessor& valueAccessor);

template <class RandomIt, typename TValueAccessor>
void placement_sort::sort(RandomIt first, RandomIt last, const TValueAccessor& valueAccessor) 
```
where valueAccessor is
```cpp
auto valueAccessor = [](const T* element) -> fund_type {return element->value};
```
or
```cpp
template <class T>
struct ValueAccessor {
    fund_type operator() (const T* element) const { return element->value;}
};

```
This to be used on structs or classes to select a field or a method to pick value from.


Example
-------

```cpp
placement_sort::sort(array.begin(), array.end(), [](const T* sportsman) -> float { return sportsman->height;});
```


Idea shortly
------------

To sort numbers 4,2,3,1 just place number i on i-th position.
If numbers range is larger or smaller than size, use shift and scale operations to fit a number in the size range. If there are collisions after shift and scale, recursively sort out all the elements which hit the same place.


Idea in details
---------------

Let's assume A is not sorted input array, B is sorted output.
1. Find min and max values in the A.
2. Define placer(x) := (x - min) * size / (max - min)
3. Move A[i] to B[placer(A[i]] for i = 1..N.

Collissions resolution.
Except cases of soring unque 1..N numbers there collisions appear. It's where place(A[i]) == place(A[j]) for some of i and j.

To detect collisions let's count elements in each final position using N counters.
If each counter[i] == 1 for any i {1..N}, then there is no collisions. Just move A[i] to B[place[i]] and stop.
Otherwise:
1. Compute memory distribution. For example if counter[1] = 1, counter[2] = 10, counter[3] = 2, then the element with place == 1 will be on the 1st position, elements with place == 2 will be on positions from 2nd to 11th, and elements with place == 3 will be on positions 12th and 13th. To save memory just replace counter[i] with index from which the corresponding intervals start. Thus counter[1] = 1, counter[2] = 2, counter[3] = 12.
2. Move elements according to the memory distribution. From the example above the element in the 1st position will be smaller than any other element. Elements in positions from 2nd to 11th will be larger than the 1st, and smaller than 12th and 13th. This could be done be moving A[i] to B[counterr[place(A[i])]] and then incrementing counter[placer(A[i])] by one, so that it points to the next place in the interval to put there the following colliding element.
3. Sort out recursively intervals with more than one element. To keep the worst case operations count limit within O(N * log(N)) the following strategy is used. If one interval is larger than N / 2, such an interval is devided into two equal subintervals, and both sorted recursively and separately, than merge sort step is used to join them.

Complexity
----------

O(n) in most cases.
Actually complexity < O(n * k), where  k = log(size, max - min) + 1 and is a number of possible recursion levels. k is typically 2 to 4 on random data.

Worst case: values like 1, N, N^2, N^3, ...,  N^N. In that case placer(A[N]) = N, and placer(A[i]) = 1 for i {1,N-1} on first step. Thus every element except one collide at one place on first recursion level. On the next recusrsion it'll repeat without last element. And there are potentially N such steps and O(N * N) operations. But due to split of intervals larger than N / 2 and merge sort fall back it'll come down only to O(N * log N).

There is 2 * N data moves one the first recursion level, and 2 * M for subintervals of size M.


Performance
-----------

Outperforms qsort on any size, std::sort(g++ 7.2.0) on N > 80, and std::stable_sort on N > 40. (as measured on Ubuntu 17.10 / Intel(R) Core(TM) i7-5820K)

[Place for chart GCC Ubuntu]

[Place for chart MSVC Windows]

Memory usage
------------

Uses `(sizeof(T) + sizeof(index_t)) * size` extra memory. 

It is possible to Non stable version can run on sizeof(index_t) * size extra memory.


Applicability
-------------

Works great to sort numerical values, and even better to sort large objects by a numerical field because of low number of data moves.
Due to the nature of soring by numerical values it's tricky to sort out data which is not numbers by it's nature, eq strings.


Possible future features
------------------------

- SMP (openmp/pthreads) 
    - sort chunks and merge sort into final

- Find only i-th to k-th elements of sorted array

- DMP
    - MPI/Distributed
        prepare blocks to send to a node 
        
    - MAP-Reduce
        Sort local chunks and merge sort on reduce


LICENSE
-------
Copyright Alexandr Kobotov 2018-2019. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
