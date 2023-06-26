#!/usr/bin/env python
# Download the SUSY dataset
#
# Copyright 2023 Markus Wallerberger
# SPDX-License-Identifier: MIT
import csv
import gzip
import io
import pathlib
import sys

import numpy as np
import requests

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"
SCRIPT_DIR = pathlib.Path(__file__).absolute().parent

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
    if progress:
        sys.stderr.write(f"Downloading and extracting ...\n{url}\n")
    rows = []
    with requests.get(URL, allow_redirects=True, stream=True) as response:
        with gzip.GzipFile(fileobj=response.raw) as csv_file:
            csv_reader = csv.reader(io.TextIOWrapper(csv_file))
            if progress:
                csv_reader = progress_bar(csv_reader, 5_000_000)
            for row in csv_reader:
                rows.append(list(map(float, row)))
    if progress:
        sys.stderr.write("Converting to array ...\n")
    return np.array(rows)


def save_dataset(dataset, fileobj):
    sys.stderr.write(f"Saving to file {fileobj.name} ...\n")
    labels = dataset[:,0]
    variables = dataset[:,1:]
    np.savez_compressed(fileobj,
                        labels=labels.astype(np.int8),
                        variables=variables.astype(np.float32),
                        variable_names=VARIABLE_NAMES)


if __name__ == "__main__":
    outpath = SCRIPT_DIR.parent / "shared" / "susy.npz"
    with open(outpath, "wb") as outfile:
        dataset = get_susy_dataset()
        save_dataset(dataset, outfile)
