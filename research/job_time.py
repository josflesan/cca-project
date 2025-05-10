from scheduler import Benchmark
import docker 
import time
from docker.models.containers import Container

ORDER = [
    Benchmark("radix", priority=4, library="splash2x", thread_num=2, cores_num=1),
    # Benchmark("freqmine", priority=3, thread_num=3, cores=["1", "2", "3"]),
    # Benchmark("canneal", priority=1, thread_num=3, cores=["1", "2", "3"]),
    # Benchmark("ferret", priority=2, thread_num=2, cores=["2", "3"]),
    # Benchmark("dedup", priority=5, thread_num=1, cores=["1"]),
    Benchmark("blackscholes", priority=6, thread_num=3, cores_num=1),
    # Benchmark("vips", priority=7, thread_num=2, cores=["2", "3"]),
]

def run_serial(client):
    lines = []
    for benchmark in ORDER:

        # Run for every thread and core number combination
        # for thread_num, core_num in [(1,1), (3,3), (2,2), (3, 2)]:
        for thread_num, core_num in [(1,1), (2,1)]:
            if benchmark.name == "radix" and thread_num == 3:
                continue

            print(f"Running {benchmark.name} with {thread_num} threads and {core_num} cores")
            benchmark.thread_num = thread_num
            benchmark.cores = ["2", "3"] if core_num == 2 else ["1", "2", "3"]
            start = time.perf_counter()
            container = client.containers.run(
                image=benchmark.image,
                command=f"./run -a run -S {benchmark.library} -p {benchmark.name} -i native -n {benchmark.thread_num}",
                name=benchmark.name,
                cpuset_cpus=f"{benchmark.cores[0]}-{benchmark.cores[-1]}",
                detach=True,
                remove=False,
            )
            container.wait()

            end = time.perf_counter()
            total_job_time = (end - start) / 60
            lines.append(f"{benchmark.name} took {total_job_time} minutes with {core_num} cores and {thread_num} threads")
            print(f"{benchmark.name} took {total_job_time} minutes with {core_num} cores and {thread_num} threads")
        
            # Cleanup container
            print(f"Killing Container: {container.name}")
            container.stop()
            client.containers.prune()

    with open("log_timings.txt", "w") as f:
        f.writelines(lines)


def test1(client):
    """
    Test running two containers at the same time
    """
    lines = []
    ORDER = [
        Benchmark("radix", priority=4, library="splash2x", thread_num=4, cores_num=1),
        Benchmark("blackscholes", priority=6, thread_num=3, cores_num=1),
    ]

    start = time.perf_counter()
    for benchmark in ORDER:
        print(f"Running {benchmark.name} with {benchmark.thread_num} threads and {benchmark.cores_num} cores")
        benchmark.thread_num = benchmark.thread_num

        if benchmark.cores_num == 1:
            cores = ["1"]
        elif benchmark.cores_num == 2:
            cores = ["1", "2"]
        else:
            cores = ["1", "2", "3"]

        container = client.containers.run(
            image=benchmark.image,
            command=f"./run -a run -S {benchmark.library} -p {benchmark.name} -i native -n {benchmark.thread_num}",
            name=benchmark.name,
            cpuset_cpus=f"{cores[0]}-{cores[-1]}",
            detach=True,
            remove=False,
        )
    
    # Wait for all containers to finish
    for container in client.containers.list():
        container.wait()

        # Cleanup container
        print(f"Killing Container: {container.name}")
        container.stop()

    end = time.perf_counter()
    total_job_time = (end - start) / 60
    lines.append(f"All benchmarks took {total_job_time} minutes")
    print(f"All benchmarks took {total_job_time} minutes")

    # Prune the containers
    client.containers.prune()

    with open("log_timings.txt", "w") as f:
        f.writelines(lines)

def main():
    client = docker.from_env()
    test1(client)

        
if __name__ == "__main__":
    main()
