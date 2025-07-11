# **bb_wdd_eval**

Tools for evaluating the performance of the Waggle Dance Detector (WDD) system.

## Installation

```bash
conda activate beesbook
pip install git+https://github.com/BioroboticsLab/bb_wdd_eval.git
```

## Available Tools

`annotation_matcher.py`: Given a spreadsheet of manually annotated waggle phases, will identify candidate matches in the automatically annotated WDD data. Matching is based on timestamp and spatial proximity within configurable tolerance thresholds.

`coordinate_transformer.py`: Contains a class for transforming coordinates between HD camera reference frame (used in manual annotations) and the WDD camera reference frame (used in automatic annotations).

## Usage

`python3 annotation_matcher.py --help`
