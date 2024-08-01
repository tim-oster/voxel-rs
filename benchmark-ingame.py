import csv
import json
import signal
import subprocess
import time


def run_benchmark(svo_type, render_distance, render_shadows):
    cmd = [
        "cargo",
        "run",
        "--release",
        "--features=benchmark," + "use-" + svo_type,
        "--no-default-features",
        "--",
        "--pos", "-644", "97", "120",
        "--rot", "-1", "165", "0",
        "--detach-input",
        "--render-distance=" + str(render_distance),
        "--fov=80",
        "--mc-world=assets/worlds/benchmark",
        "--render-shadows=" + ("true" if render_shadows else "false"),
    ]

    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)

    while True:
        output = process.stdout.readline().rstrip().decode("utf-8")
        if output == "" and process.poll() is not None:
            break
        if output == "all chunks loaded":
            break

    time.sleep(10)

    process.send_signal(signal.CTRL_BREAK_EVENT)
    process.wait()

    try:
        output, _ = process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.terminate()
        output, _ = process.communicate()

    for line in output.decode("utf-8").split("\n"):
        prefix = "benchmark: "
        if not line.startswith(prefix):
            continue
        return json.loads(line.lstrip(prefix))

    return None


def unique_combinations(matrix):
    results = []

    for key, values in matrix.items():
        if len(results) == 0:
            for v in values:
                results.append({key: v})
            continue

        new_results = []
        for v in values:
            for r in results:
                copy = r.copy()
                copy[key] = v
                new_results.append(copy)
        results = new_results

    return results


def nested_keys(d: dict, path=""):
    for k, v in d.items():
        if type(v) is dict:
            yield from nested_keys(v, path=path + k + ".")
        else:
            yield path + k, v


def main():
    matrix = {
        "render_distance": [10, 20, 30, 40],
        "render_shadows": [True, False],
        "passes": list(range(4)),
        # put svo type last so that build cache is only discarded once
        "svo_type": ["esvo", "csvo"],
    }

    results = []
    for i, case in enumerate(unique_combinations(matrix)):
        print("running case", case)
        r = run_benchmark(case["svo_type"], case["render_distance"], case["render_shadows"])
        results.append((case, r))

    if len(results) == 0:
        print("no results")
        return

    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)

        header = [k for k in matrix.keys()]
        header.extend(dict(nested_keys(results[0][1])).keys())
        w.writerow(header)

        for r in results:
            row = [v for v in r[0].values()]
            row.extend(dict(nested_keys(r[1])).values())
            w.writerow(row)


if __name__ == "__main__":
    main()
