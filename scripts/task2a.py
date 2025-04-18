from pathlib import Path
import subprocess
import re
import argparse
import tqdm


def extract_real_time(text):
    match = re.search(r"real\s+(\d+)m([\d.]+)s", text)

    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        total_seconds = minutes * 60 + seconds
        return total_seconds

    return None


def get_logs_cmd(job_name: str):
    return (
        f"kubectl logs $(kubectl get pods --selector=job-name={job_name}"
        + " --output=jsonpath='{.items[*].metadata.name}')"
    )


def wait_for_job_completion(job_name: str, timeout_seconds=3600):
    """Wait for the Kubernetes job to complete."""
    wait_cmd = f"kubectl wait --for=condition=complete job/{job_name} --timeout={timeout_seconds}s"
    result = subprocess.run(wait_cmd, shell=True, text=False)
    return result.returncode == 0


def run_benchmarks(interference: str):
    part2a_dir = Path("parsec-benchmarks/part2a")
    fila_name = f"logs/part2a/log_{interference}.txt"
    for program_file in (t := tqdm.tqdm(list(part2a_dir.iterdir()))):
        job_name, _ = program_file.name.split(".")
        t.set_description(f"running {job_name}")
        # job_name = f"parsec-{program}"
        # print(program)

        start_job_cmd = f"kubectl create -f {program_file}"
        _ = subprocess.run(start_job_cmd, text=False, shell=True)

        wait_for_job_completion(job_name)

        logs = subprocess.run(
            get_logs_cmd(job_name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            shell=True,
        )
        real_time = extract_real_time(logs.stdout)
        with open(fila_name, "a") as f:
            f.write(f"{job_name}: {real_time} \n")

    # Cleanup
    subprocess.run("kubectl delete jobs --all", shell=True)
    subprocess.run("kubectl delete pods --all", shell=True)


def calculate_normalized_time(interference: str):
    # Read baseline
    with open("logs/part2a/log_None.txt", "r") as f:
        baseline = f.readlines()

    # Read interference file
    with open(f"logs/part2a/log_{interference}.txt", "r") as f:
        interference_logs = f.readlines()

        for interf_line, baseline_line in zip(interference_logs, baseline):
            benchmark = interf_line.split(": ")[0]
            baseline_seconds = float(baseline_line.split(": ")[1])
            interference_seconds = float(interf_line.split(": ")[1])

            normalized_time = round(interference_seconds / baseline_seconds, 2)
            color = "🧧"
            if normalized_time <= 1.3:
                color = "💚"
            elif normalized_time > 1.3 and normalized_time <= 2:
                color = "🔶"

            print(f"{benchmark}: {normalized_time} {color}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    args = parser.parse_args()

    # run_benchmarks(args.i)
    # print()
    calculate_normalized_time(args.i)
