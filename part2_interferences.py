from pathlib import Path
import subprocess
import shlex
import re
import uuid
import argparse
import tqdm

def extract_real_time(text):
    match = re.search(r'real\s+(\d+)m([\d.]+)s', text)
    return match

def get_logs_cmd(job_name: str):
    return (
        f"kubectl logs $(kubectl get pods --selector=job-name={job_name}"
        + " --output=jsonpath='{.items[*].metadata.name}')"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    args = parser.parse_args()
    part2a_dir = Path("parsec-benchmarks/part2a")
    fila_name = f"log_{args.i}.txt"
    for program_file in (t := tqdm.tqdm(part2a_dir.iterdir())):

        job_name, _ = program_file.name.split(".")
        t.set_description(f"running {job_name}")
        # job_name = f"parsec-{program}"
        # print(program)

        start_job_cmd = f"kubectl create -f {program_file}"
        _ = subprocess.run(start_job_cmd, text=True, shell=True)
        logs = subprocess.run(
            get_logs_cmd(job_name),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            shell=True,
        )
        real_time = extract_real_time(logs)
        with open(fila_name, "a")  as f:
            f.write(f"{job_name}: {real_time} \n")


if __name__ == "__main__":
    main()
