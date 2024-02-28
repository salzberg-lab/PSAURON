PSAURON
===========

![License](https://img.shields.io/badge/license-MIT-blue.svg "License")
[![Pypi Release](https://badge.fury.io/py/psauron.svg)](https://pypi.org/project/psauron/)
[![codecov](https://codecov.io/github/salzberg-lab/PSAURON/graph/badge.svg?token=GP88IZQFKA)](https://codecov.io/github/salzberg-lab/PSAURON)

**PSAURON** is a machine learning model for rapid assessment of protein coding gene annotation. Link to paper coming soon...

Installation
------------------

```
$ pip install psauron
```

PSAURON can run on GPU or CPU and depends on PyTorch, which can be annoying :disappointed:

It may help to install PSAURON in a virtual enviromment :slightly_smiling_face:
```
$ python3 -m venv /path/to/new/virtual/environment
$ source /path/to/new/virtual/environment/bin/activate
$ pip install psauron
```

Quickstart
------------------

PSAURON takes as input a single multi-fasta file and outputs a .csv with scores for all reading frames.

By default, PSAURON uses all six frames of the nucleotide coding sequences (CDS).
```
$ psauron -i path_to_your_CDS.fa -o path_to_output.csv
```

You may also provide a multi-fasta with protein (amino acid) sequence.
```
$ psauron -i path_to_your_protein.faa -o path_to_output.csv -p 
```

...or request PSAURON score only the in-frame nucleotide sequence.
```
$ psauron -i path_to_your_CDS.fa -o path_to_output.csv -s
```

Note: internal stop codons are ignored by PSAURON. A high PSAURON score does not guarantee a sequence contains a valid ORF. This is intended behavior, as alternate frame scores are used by default to boost the power of the model. 

Usage
------------------
```
psauron [-h] -i INPUT_FASTA [-o OUTPUT_PATH] [-m MINIMUM_LENGTH] [-e EXCLUDE] [--inframe INFRAME] [--outframe OUTFRAME] [-c] [-s] [-p] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_FASTA, --input-fasta INPUT_FASTA
                        REQUIRED path to FASTA with spliced CDS sequence or protein sequence. A spliced CDS fasta can be created from a GTF/GFF and a reference FASTA by using gffread.
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        OPTIONAL path to output results file, default=./psauron_score.csv
  -m MINIMUM_LENGTH, --minimum-length MINIMUM_LENGTH
                        OPTIONAL exclude all proteins shorter than m amino acids, default=5
  -e EXCLUDE, --exclude EXCLUDE
                        OPTIONAL exclude any CDS where FASTA description contains given text (case invariant), e.g. "hypothetical", default=None
  --inframe INFRAME     OPTIONAL probability threshold used to determine final psauron score, in-frame, higher number decreases sensitivity and increases specificity, default=0.5, range=[0,1]
  --outframe OUTFRAME   OPTIONAL probability threshold used to determine final psauron score, out-of-frame, higher number increases sensitivity and decreases specificity, default=0.5, range=[0,1]
  -c, --use-cpu         OPTIONAL set -c to force usage of CPU instead of GPU, default=False
  -s, --single-frame    OPTIONAL set -s to score only the in-frame CDS, which may lower accuracy of the model, default=False
  -p, --protein         OPTIONAL set -p if your FASTA contains amino acid protein sequence, which may lower accuracy of the model, default=False
  -v, --verbose         OPTIONAL set -v for verbose output with progress bars etc., default=False

 -i INPUT_FASTA, REQUIRED path to FASTA with spliced CDS sequence. This fasta can be created from a GTF/GFF and a reference FASTA by using gffread.
```

Example gffread commands to get CDS FASTA:
```
gffread -x CDS_FASTA.fa -g genome.fa input.gff
gffread -x CDS_FASTA.fa -g genome.fa input.gtf
```
