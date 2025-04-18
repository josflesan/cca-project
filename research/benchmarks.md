#### Nodes

The nodes we have in the cluster:

1. `node-a-2core`: instance of e2-highmem-2 (16GB RAM, 2.25GHz) VM so suitable for loads that require a lot of memory and can run in single core environments ("N2 high-memory machine types have 8 GB of system memory per vCPU." - Google)
2. `node-b-2core`: instance of n2-highcpu-2 (2GB RAM, 2.6GHz) VM so good for loads that require more processing power and less memory but do not benefit from parallelisation as much
3. `node-c-4core`: instance of c3-highcpu-4 (8GB RAM, 1.9GHz),so good for loads that require more computing power and also benefit from parallelization
4. `node-d-4core`: instance of n2-standard-4 (16GB RAM, 2.6GHz) so good for loads that require more computing power, benefit from parallelization and are more memory hungry

---

#### Ordering

Ordering (Shortes Job First):

dedup, radix, vips, blackscholes, canneal, ferret, freqmine

---

#### Assignments

The benchmarks have the following properties. We should collocate processes that are memory-hungry with others that aren't as affected by memory interference (and same for everything):

Processes in A:

1) *canneal*: performs simulated annealing to optimize routing cost in chip design. Parallel workload that seems to not benefit as much from parallelism. Instead it is most affected by memory bandwidth so we probably want to use a highmem instance (A). T = 4

Processes in B:

1) *dedup*: data compression algorithm. Does not benefit much from parallelism but is affected by CPU and memory (B). Run it once and then leave this node available

1) *blackscholes*: using Black-scholes model to compute options. Data-parallel/Embarrassingly parallel workload, should benefit from parallelism in multicore system, probably does not need as much memory (C/B?)

2) *vips*: image processing, pipeline parallel, memory hungry. Benefits moderately from parallelism (B). T = 4, anything else is not helping

Processes in C:

1) *freqmine*: data mining using recursive data structures, very memory intensive. Data parallel so it also benefits from multithreading (C). Might be an idea to keep this isolated so it does not have to compete with other jobs for resources. T = 4/8

Processes in D:

1) *radix*: Heavily benefits from parallelism. Can be collocated with other parallel and data-hungry processes (D). Maybe pin this to a separate core?

1) *ferret*: content-based similarity search for multimedia files. Pipeline parallelism so a lot of data exchange between stages. Benefits a lot from parallelism and needs high memory VM so maybe standard (D). T = 4/8

---

#### Notes

We set the requests and limits even for jobs that are not collocating to guarantee predictable performance.
