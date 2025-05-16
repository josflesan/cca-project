import re
import argparse
import subprocess

from launch_cluster_part4 import update_bash_script, get_internal_ip, get_instance_name

def update_config(config: str, internal_ip: str | None, threads: int=1) -> str:
    # Replace memory limit (-m) with 1024
    config = re.sub(r"^-m\s+\d+", "-m 1024", config, flags=re.MULTILINE)

    # Replace listen address (-l) with internal IP
    if internal_ip:
        config = re.sub(r"^-l\s+\S+", f"-l {internal_ip}", config, flags=re.MULTILINE)

    # Set number of threads (-t); add if not present
    if re.search(r"^-t\s+\d+", config, flags=re.MULTILINE):
        config = re.sub(r"^-t\s+\d+", f"-t {threads}", config, flags=re.MULTILINE)
    else:
        config += f"\n-t {threads}\n"
    
    return config

def launch_memcached(
    instance: str, 
    threads: int, 
    internal_ip: str,
    coreStart: int,
    coreEnd: int
):
    # Expose the service
    subprocess.run(f"gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing ubuntu@{instance}:/etc/memcached.conf . --zone europe-west1-b", shell=True)
    with open("memcached.conf", "r") as f:
        config = f.read()
    
    updated_config = update_config(config, internal_ip=internal_ip, threads=threads)
    
    with open("memcached.conf", "w") as f:
        f.write(updated_config)

    # 1. Copy to home dir
    subprocess.run(
        f"gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing memcached.conf ubuntu@{instance}:~/memcached.conf --zone europe-west1-b", shell=True, check=True)

    # 2. Move with sudo
    subprocess.run(
        f"gcloud compute ssh ubuntu@{instance} --zone europe-west1-b --ssh-key-file ~/.ssh/cloud-computing --command 'sudo mv ~/memcached.conf /etc/memcached.conf'", shell=True, check=True)

    # Restart the service
    update_bash_script("scripts/setup/restart_service.bash", coreStart=coreStart, coreEnd=coreEnd)
    subprocess.run(f"gcloud compute scp --ssh-key-file ~/.ssh/cloud-computing scripts/setup/restart_service.bash ubuntu@{instance}:~/ --zone europe-west1-b", shell=True)
    subprocess.run(f"gcloud compute ssh --ssh-key-file ~/.ssh/cloud-computing ubuntu@{instance} --zone europe-west1-b --command='bash ~/$(basename scripts/setup/restart_service.bash)'", shell=True)
    subprocess.run("exit", shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments to Launch Cluster on GCloud")
    parser.add_argument("--threads", type=int, help="Number of threads to launch memcached with")
    parser.add_argument("--core_start", type=int, help="Beginning of Core ID to launch memcached on")
    parser.add_argument("--core_end", type=int, help="End of Core ID to launch memcached on")
    args = parser.parse_args()

    # Get instance names and internal IPs
    server = get_instance_name("memcache-server")
    internal_server_ip = get_internal_ip(server)

    # Launch memcached service
    launch_memcached(server, args.threads, internal_server_ip, args.core_start, args.core_end)
