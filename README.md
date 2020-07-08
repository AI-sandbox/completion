# Completion
Iterative SVD Completion method for ancestry-specific dimensionality reduction.

## Usage
Run the method using the following command from command line.
```bash
python3 iterative_svd_method.py params.txt
```
`params.txt` is the parameters file that is passed as input to the method. The following parameters can be specified in the parameters file:
* `BEAGLE_OR_VCF` (int): `1` if the genetic data file is a Beagle file, or `2` if it is a VCF file.
* `BEAGLE_FILE` (str): path to the Beagle file if the genetic data file is a Beagle file.
* `VCF_FILE` (str): path to the VCF file if the genetic data file is a VCF file.
* `IS_MASKED` (bool): `True` if an ancestry file is passed for ancestry-specific masking, or `False` otherwise.
* `VIT_OR_FBK_OR_TSV` (int): `1` if the ancestry file is a VIT file, `2` if it is an FBK file, or `3` if it is a TSV file.
* `VIT_FILE` (str): path to the VIT file if the ancestry file is a VIT file.
* `FBK_FILE` (str): path to the FBK file if the ancestry file is a FBK file.
* `FB_OR_MSP` (int): `1` if the TSV ancestry file is an FB file, or `2` if it is an MSP file.
* `TSV_FILE` (str): path to the TSV file if the ancestry file is a TSV file.
* `NUM_ANCESTRIES` (int): the total number of ancestries in the ancestry file.
* `ANCESTRY` (int): ancestry number of the ancestry for which dimensionality reduction is to be performed. Ancestry counter starts at 0 if the ancestry file is a TSV file, and starts at 1 if it is a VIT or an FBK file.
* `PROB_THRESH` (float): minimum probability threshold for a SNP to belong to an ancestry, if the ancestry file is an FBK file or an FB TSV file.
* `AVERAGE_PARENTS` (bool): `True` if the DNAs from the two parents are to be combined (averaged) for each individual, or `False` otherwise.
* `START_RANK` (int): smallest rank among the range of ranks, from which the best rank is to be chosen for Iterative SVD using cross-validation.
* `END_RANK` (int) = largest rank among the range of ranks, from which the best rank is to be chosen for Iterative SVD using cross-validation.
* `RANK` (int): rank for Iterative SVD if no cross-validation is to be performed.
* `CHOOSE_BEST` (bool): `True` if cross-validation is to be performed to choose the best rank for Iterative SVD, or `False` otherwise.
* `NUM_CORES` (int): number of cores to use for performing cross-validation in parallel.
* `IS_WEIGHTED` (bool): `True` if weights are provided in the labels file, or `False` otherwise.  
* `LABELS_FILE` (str): path to the labels file. It should be a TSV file where the first column has header `indID` and contains the individual IDs, and the second column has header `label` and contains the labels for all individuals. If `IS_WEIGHTED` is specified as `True`, then the file must have a third column that has header `weight` and contains the weights for all individuals.
NOTE: Individuals with zero weight are removed. Negative weights are used to combine individuals and replace them with a single average individual. Provide a weight of `-1` to the first set of individuals to be combined, `-2` to the second set of individuals to be combined, and so on. Each set of individuals that is to be combined must have the same label.
* `OUTPUT_FILE` (str): path to the output file, to which the output of the run is written. It is a TSV file with 3 columns. The first column contains the individual IDs, and the second and third column contain the ancestry-specific projections obtained after dimensionality reduction.
* `SCATTERPLOT_FILE` (str): path to the scatter plot file with `.html` extension. The scatter plot of the individuals is saved in this file.
* `SAVE_MASKED_MATRIX` (bool): `True` if the masked matrix is to be saved as a binary file, or `False` otherwise.
* `MASKED_MATRIX_FILE` (str): path to the masked matrix file. The masked matrix is saved in this file.
* `SAVE_COMPLETED_MATRIX` (bool): `True` if the completed matrix is to be saved as a binary file, or `False` otherwise.
* `COMPLETED_MATRIX_FILE` (str): path to the completed matrix file. The completed matrix is saved in this file.

NOTE: The parameters file must have all the above parameters. Each line in the parameters file must have a parameter name followed by `=`, followed by the value for that parameter. The value for a parameter that is not useful for the run can be filled with any value compatible with the parameter type.

NOTE: There are 2 acceptable formats for SNP indices in the Beagle file: 
1. rsid: `rs` followed by the id (integer). For example, `rs12345`.
2. position: chromosome number (integer) followed by `_`, followed by the position (integer). For example, `10_12345`.