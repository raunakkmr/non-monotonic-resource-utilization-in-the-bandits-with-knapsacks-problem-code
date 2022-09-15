# Non-monotonic Resource Utilization in the Bandits with Knapsacks Problem
This repository contains code for the paper "Non-monotonic Resource Utilization in the Bandits with Knapsacks Problem".

## Usage

1. Create a data file according to the format mentioned in the `parse_datafile` function in `src/data.py` and store it in the `data` directory.
2. Change directory to `src`.
3. Run `python main.py --data_filename {name of the data file created in step 1} --trials {number of trials of the algorithms to average over}` --learning {include if you want to run the learning algorithm, exclude if not}.
4. The plots are stored in the `plots` directory.

## Citation

If you use this code, please cite the following paper
```
@inproceedings{kumar2022nonmonotonic,
  author    = "Kumar, Raunak and Kleinberg, Robert D.",
  title     = "Non-monotonic Resource Utilization in the Bandits with Knapsacks Problem",
  year      = "2022",
  booktitle = "Proceedings of the 36th Conference on Neural Information Processing Systems (\textbf{NeurIPS})"
}
```
