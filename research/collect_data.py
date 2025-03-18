from pathlib import Path
import csv
import subprocess
import re
import tqdm
_NUM_RUNS = 3

in_path = Path("../interference")
out_path = Path("../data")


def save_output(out: str, run: int, benchmark: str) -> None:
    file_path = out_path / f"{benchmark}_out_{run}.csv"
    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header, *lines = out.split("\n")
        header = header[1:].strip()
        header = re.split('\s+', header)
        writer.writerow(header)
        for line in lines:
            if not line:
                break
            else:
                line = line.strip()
                fields = re.split(r'\s+', line)
                writer.writerow(fields)

def main() -> None:
    path_iter = tqdm.tqdm(in_path.iterdir())
    for file_path in path_iter:
        start_job_cmd = f"kubectl create -f interference/{file_path.name}"
        job_name, _ = file_path.name.split(".")
        delete_job_cmd = f"kubectl delete pods {job_name}"
        for run_no in range(_NUM_RUNS):
            path_iter.set_description(f"running file={file_path.name} run={run_no}")
            result = subprocess.run(start_job_cmd,stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, shell=True)
            #! is there something to do with kubectl get pods -o wide?
            save_output(result.stdout, run=run_no, benchmark=job_name)
            _ = subprocess.run(delete_job_cmd, text=True, shell=True)


if __name__ == "__main__":
    main()
