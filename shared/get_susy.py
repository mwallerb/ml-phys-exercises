import csv
import gzip
import io
import sys

import numpy as np
import requests

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"

VARIABLE_NAMES = [
    "lepton 1 pT",
    "lepton 1 eta",
    "lepton 1 phi",
    "lepton 2 pT",
    "lepton 2 eta",
    "lepton 2 phi",
    "missing energy magnitude",
    "missing energy phi",
    "MET_rel",
    "axial MET",
    "M_R",
    "M_TR_2",
    "R",
    "MT2",
    "S_R",
    "M_Delta_R",
    "dPhi_r_b",
    "cos(theta_r1)"
    ]


def progress_bar(inner, total):
    step = total // 80
    for i, item in enumerate(inner):
        if not (i % step):
            sys.stderr.write("\r[" + ("@" * (i // step)).ljust(80, ".") + "]")
        yield item
    sys.stderr.write("\r[" + "@" * 80 + "]\n")


def get_susy_dataset(url=URL, progress=True):
    # Be kind: do not execute this many times, as it downloads a large file
    # from a university server.
    sys.stderr.write("Downloading file... ")
    rows = []
    with requests.get(URL, allow_redirects=True) as response:
        sys.stderr.write("Got file.\n")
        with io.BytesIO(response.content) as response_file:
            with gzip.GzipFile(fileobj=response_file) as csv_file:
                csv_reader = csv.reader(io.TextIOWrapper(csv_file))
                if progress:
                    csv_reader = progress_bar(csv_reader, 5_000_000)
                for row in csv_reader:
                    rows.append(list(map(float, row)))
    return np.array(rows)


def save_dataset(dataset, filename="susy.npz"):
    labels = dataset[:,0]
    variables = dataset[:,1:]
    np.savez_compressed("../shared/susy.npz",
                        labels=labels.astype(np.int8),
                        variables=variables.astype(np.float32),
                        variable_names=VARIABLE_NAMES)


if __name__ == "__main__":
    dataset = get_susy_dataset()
    save_dataset(dataset)
