import time as t
import sys 
import subprocess
import statistics as stats
from tqdm.auto import tqdm
import json
import argparse


def run_once(cmd):
    start = t.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = t.perf_counter()
    return end - start

def benchmark(cmd, warmups = 2, repeats = 5):
    for _ in range(warmups):
        run_once(cmd)

    times = []
    for runs, _ in tqdm(enumerate(range(repeats)), total=len(range(repeats)), desc="runs"):
        run_time = run_once(cmd)
        times.append(run_time)
    
    mean = stats.mean(times)
    std = stats.stdev(times)

    return times, mean, std


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot 2D Ising Model with external field')
    parser.add_argument("--L", type=int, default=32,
                        help="Lattice size")
    args = parser.parse_args()
    
    L = args.L

    common_flags = ['--L', str(L)]

    metropolis_cmd = [sys.executable, 'src/sim.py'] + common_flags
    hb_cmd = [sys.executable, 'src/heat_bath.py'] + common_flags

    print('Benchmarking metropolis...')
    times_met, mean_met, std_met = benchmark(metropolis_cmd)

    print('Benchmarking Heat Bath....')
    times_hb, mean_hb, std_hb = benchmark(hb_cmd)

    print(f'\n----Results (L = {L})----')
    print(f'Metropolis: {mean_met:.3f} ± {std_met:.3f}')
    print(f'Heat Bath: {mean_hb:.3f} ± {std_hb:.3f}')


    speedup = mean_met/ mean_hb

    print(f'\nSpeedup (Metropolis / Heat bath) = {speedup:.3f}x')

    results = {
        'mean_met': mean_met,
        'mean_hb': mean_hb,
        'std_met': std_met,
        'std_hb': std_hb,
        'times_met': times_met,
        'times_hb':times_hb
    }

    with open(f"./data/benchmark_L{L}.json", "w") as f:
            json.dump(results, f, indent=4)

    print('Results saved in .json')



