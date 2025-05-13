import docker
import time
from docker.models.containers import Container
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class Job(Enum):
    SCHEDULER = "scheduler"
    MEMCACHED = "memcached"
    BLACKSCHOLES = "blackscholes"
    CANNEAL = "canneal"
    DEDUP = "dedup"
    FERRET = "ferret"
    FREQMINE = "freqmine"
    RADIX = "radix"
    VIPS = "vips"


@dataclass
class Benchmark:
    name: str
    priority: int = 0
    thread_num: int = 1
    cores_num: int = 1
    container: Container | None = None
    library: str = "parsec"

    def __post_init__(self):
        if self.name == "radix":
            self.image: str = "anakli/cca:splash2x_radix"
            self.library: str = "splash2x"
        else:
            self.image: str = f"anakli/cca:parsec_{self.name}"

    def to_job(self) -> Job:
        return Job._member_map_[self.name.upper()]

    def attach_container(self, container: Container) -> None:
        self.container = container

    def get_cores(self) -> List[int]:
        core_start, core_end = self.container.attrs["HostConfig"]["CpusetCpus"].split(
            "-"
        )
        return list(range(int(core_start), int(core_end) + 1))

    def is_paused(self):
        # Maybe we have to reload, dont think so tho
        return self.container and self.container.status == "paused"

    def __str__(self):
        return f"(name={self.name}, p={self.priority})"

    def __repr__(self):
        return f"(name={self.name}, p={self.priority})"


@dataclass
class Experiment:
    name: str
    benchmarks: List[Tuple[Benchmark, List[int]]]
    killTime: int | None = None


# CRUCIAL ONES: ferret, freqmine, canneal, vips

ferret = Benchmark("ferret", priority=1, thread_num=3, cores_num=3)
freqmine = Benchmark("freqmine", priority=2, thread_num=3, cores_num=3)
canneal = Benchmark("canneal", priority=3, thread_num=2, cores_num=2)
dedup = Benchmark("dedup", priority=3, thread_num=1, cores_num=1)
radix = Benchmark("radix", priority=3, thread_num=4, cores_num=1, library="splash2x")
blackscholes = Benchmark("blackscholes", priority=3, thread_num=3, cores_num=1)
vips = Benchmark("vips", priority=4, thread_num=3, cores_num=2)


combinations = [
    Experiment(name="Radix + Dedup", benchmarks=[(radix, [1]), (dedup, [1])]),
    Experiment(name="Canneal + Radix", benchmarks=[(canneal, [2, 3]), (radix, [2])]),
    Experiment(
        name="Blackscholes + Radix", benchmarks=[(blackscholes, [2, 3]), (radix, [2])]
    ),
    Experiment(
        name="Blackscholes + Radix", benchmarks=[(blackscholes, [2]), (radix, [2])]
    ),
    Experiment(
        name="Blackscholes + Canneal",
        benchmarks=[(blackscholes, [2, 3]), (canneal, [2, 3])],
    ),
    Experiment(
        name="Ferret + Canneal",
        benchmarks=[(ferret, [1, 2, 3]), (canneal, [2, 3])],
        killTime=500,
    ),  #!!! (173 + 222) = 395
    Experiment(
        name="Ferret + Vips",
        benchmarks=[(ferret, [1, 2, 3]), (vips, [2, 3])],
        killTime=400,
    ),  #!!! (173 + 114) = 287
    Experiment(
        name="Freqmine + Canneal",
        benchmarks=[(freqmine, [1, 2, 3]), (canneal, [2, 3])],
        killTime=300,
    ),  #!!!
    Experiment(
        name="Radix + Dedup + Blackscholes",
        benchmarks=[(radix, [2]), (dedup, [2]), (blackscholes, [2])],
    ),  # Gain probably small but worth a shot
]


def test_benchmark_with_memcached(client: docker.DockerClient, bench: Benchmark, memcached_cores: int):
    """Can we collocate blackscholes and/or canneal with memcached?"""

    start = time.perf_counter()

    pin_to_cores = "1"
    container = client.containers.run(
        image=bench.image,
        command=f"./run -a run -S {bench.library} -p {bench.name} -i native -n {bench.thread_num}",
        name=bench.name,
        cpuset_cpus=pin_to_cores,
        detach=True,
        remove=False,
    )
    container.wait()
    end = time.perf_counter()

    print(f"Time Taken to run {bench.name} on memcached: {((end - start) / 60):.3f}")

def test_vips_container_creation(client: docker.DockerClient):
    """How long does VIPS take to run"""

    benchmark = vips
    start = time.perf_counter()

    client.containers.create(
        image=benchmark.image,
        command=f"./run -a run -S {benchmark.library} -p {benchmark.name} -i native -n {benchmark.thread_num}",
        name=benchmark.name,
        cpuset_cpus="2-3",
        detach=True,
        auto_remove=False
    )

    end = time.perf_counter()

    print(f"Time Taken to create VIPS: {((end - start) / 60):.3f}")


def test_colocated(client: docker.DockerClient, exp: Experiment):
    """Test collocating containers in the same cores"""

    print(f"Starting Experiment: {exp.name}")
    start = time.perf_counter()

    for benchmark, cores in exp.benchmarks:
        print(f"Running {benchmark.name} on cores {cores}...")

        container = client.containers.run(
            image=benchmark.image,
            command=f"./run -a run -S {benchmark.library} -p {benchmark.name} -i native -n {benchmark.thread_num}",
            name=benchmark.name,
            cpuset_cpus=f"{cores[0]}-{cores[-1]}",
            detach=True,
            remove=False,
        )

    while True:
        # Reload containers to update status
        for container in client.containers.list():
            container.reload()

        # If the containers have finished, end
        if all(
            [container.status == "exited" for container in client.containers.list()]
        ):
            print("ALL BENCHMARKS DONE!")
            break

        # If we are passed the killTime, end it
        if exp.killTime and (time.perf_counter() - start) >= exp.killTime:
            print(f"EXPERIMENT TIMED OUT, TERMINATING {exp}")
            break

    end = time.perf_counter()
    total_job_time = (end - start) / 60
    print(f"All benchmarks took {total_job_time} minutes")

    # Cleanup the containers
    for container in client.containers.list():
        print(f"Killing Container: {container.name}")
        container.stop()

    client.containers.prune()


def main():
    client = docker.from_env()

    # for experiment in combinations:
    #     test_colocated(client, experiment)

    # test_vips_container_creation(client)

    test_benchmark_with_memcached(client, canneal, 2)


if __name__ == "__main__":
    main()


"""
RESULTS

Starting Experiment: Radix + Dedup
Running radix on cores [1]...
Running dedup on cores [1]...
All benchmarks took 1.1382552094500018 minutes

Starting Experiment: Canneal + Radix
Running canneal on cores [2, 3]...
Running radix on cores [2]...
All benchmarks took 2.3729579399999996 minutes (vs 2.04)

Starting Experiment: Blackscholes + Radix
Running blackscholes on cores [2, 3]...
Running radix on cores [2]...
All benchmarks took 1.3027288516499993 minutes

Starting Experiment: Blackscholes + Radix
Running blackscholes on cores [2]...
Running radix on cores [2]...
All benchmarks took 2.4142674865666627 minutes

Starting Experiment: Blackscholes + Canneal
Running blackscholes on cores [2, 3]...
Running canneal on cores [2, 3]...
All benchmarks took 2.82346696886666 minutes

Starting Experiment: Ferret + Canneal
Running ferret on cores [1, 2, 3]...
Running canneal on cores [2, 3]...
All benchmarks took 4.30492017975001 minutes

# Starting Experiment: Ferret + Vips
# Running ferret on cores [1, 2, 3]...
# Running vips on cores [2, 3]...
# All benchmarks took 3.388928733900002 minutes

Starting Experiment: Freqmine + Canneal
Running freqmine on cores [1, 2, 3]...
Running canneal on cores [2, 3]...
All benchmarks took 4.729047462650002 minutes

Starting Experiment: Radix + Dedup + Blackscholes
Running radix on cores [2]...
Running dedup on cores [2]...
Running blackscholes on cores [2]...
All benchmarks took 2.7888151454333334 minutes


Conclusions:
Radix + dedup, radix takes .35 and dedup .35 where as colocating them takes 1.13
Radix + canneal,

"""
