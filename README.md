# fsl_randomise

FSL randomise related scripts.



## TODO
- Write a complete test for randomise_summary.py
- Write up contrast line translation functions for interaction effect
- ADD atlas query output to the html summary




## Scripts
1. `randomise_parallel_pnl` : runs FSL randomise in parallel using bsub
2. `randomise_summary.py` : used to summarize FSL randomise outputs


---


## 1. `randomise_parallel_pnl`

Dispatches `randomise_parallel` splited jobs through `bsub`.


### Dependencies

```
FSL
bsub
```


### Usage
```sh
# same as fsl randomise
randomise_parallel_pnl -i <4D_input_data> -o <output_rootname> -d design.mat -t design.con -m <mask_image> -n 5000 -T2
```



## 2. `randomise_summary.py`

Summarize outputs from FSL randomise. 
- design matrix
- design contrast
- p-values
- percentage of significant voxels (number of significant voxels / total number of voxels in the skeleton)
- extract values for each subject in the significant cluster
- create html summary


### Dependencies

```
FSL
python 3
nibabel
pandas
numpy
tabulate
Path
tempfile
nifti_snapshot 
jinja2
os
getpass
getpwuid
```


### Usage

#### Simple use

```sh
# path where tbss_*corrp_tstat*nii.gz
cd /TBSS/STAT/DIR
randomise_summary.py 
randomise_summary.py --constrat design.con --matrix design.mat
randomise_summary.py --input tbss_FA_corrp_tstat1.nii.gz
```

#### More advanced use

```sh
randomise_summary.py --sig_only
randomise_summary.py --figure
randomise_summary.py --atlasquery
randomise_summary.py --tbss_fill 
randomise_summary.py --subject_values
randomise_summary.py --skeleton_summary 
randomise_summary.py --skeleton_summary --tbss_all_loc ${tbss_all_out_dir} --html_summary
```

#### What I would use

```sh
randomise_summary.py 
randomise_summary.py \
    --figure \
    --tbss_fill \
    --subject_values \
    --skeleton_summary \
    --tbss_all_loc /TBSS/ALL/DIR \
    --html_summary
google-chrome randomise_summary.html
```


### Example outputs

```sh
randomise_summary.py
```

```bash
Importing modules
Importing modules complete

--------------------------------------------------------------------------------
* Summarizing information for files below
--------------------------------------------------------------------------------
	/data/pnl/kcho/tbss_example/enigma-tbss/stats/precompute_1_randomise/tbss_FW_tfce_corrp_tstat1.nii.gz
	/data/pnl/kcho/tbss_example/enigma-tbss/stats/precompute_1_randomise/tbss_FW_tfce_corrp_tstat2.nii.gz
	/data/pnl/kcho/tbss_example/enigma-tbss/stats/precompute_1_randomise/tbss_FAt_tfce_corrp_tstat1.nii.gz
	/data/pnl/kcho/tbss_example/enigma-tbss/stats/precompute_1_randomise/tbss_FAt_tfce_corrp_tstat2.nii.gz
	/data/pnl/kcho/tbss_example/enigma-tbss/stats/precompute_1_randomise/tbss_FA_tfce_corrp_tstat1.nii.gz
	/data/pnl/kcho/tbss_example/enigma-tbss/stats/precompute_1_randomise/tbss_FA_tfce_corrp_tstat2.nii.gz

--------------------------------------------------------------------------------
* Matrix summary
--------------------------------------------------------------------------------
Contrast file : design.con
Matrix file : design.mat

total number of data point : 135
Group columns are : col 1, col 2
+---------------+---------+---------+--------------------------------------------------------------------------+
|               | col 1   | col 2   | col 3                                                                    |
|---------------+---------+---------+--------------------------------------------------------------------------|
| mean          | 0.39    | 0.61    | 35.81                                                                    |
| std           | 0.49    | 0.49    | 11.21                                                                    |
| min           | 0.0     | 0.0     | 18.0                                                                     |
| max           | 1.0     | 1.0     | 63.0                                                                     |
| unique values | 0. 1.   | 0. 1.   | 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.  |
|               |         |         |  36. 37. 38. 39. 40. 41. 42. 43. 44. 45. 46. 47. 49. 51. 52. 53. 55. 56. |
|               |         |         |  57. 59. 60. 63.                                                         |
| count         | 53      | 82      | 2  2  3  7  1  4  3  7  3  2  5 11  3  3  4  3  2  6  6  4  3  7  2  2   |
|               |         |         |   2  4  5  2  3  3  1  4  2  2  2  4  2  1  1  2                         |
| column info   | Group 1 | Group 2 | nan                                                                      |
+---------------+---------+---------+--------------------------------------------------------------------------+


--------------------------------------------------------------------------------
* Result summary
--------------------------------------------------------------------------------
+-----------------------------------+-------------+-------------------+--------+------------+----------------+-----------+-------------+--------------+------------------------+----------+-----------+
| file name                         | contrast    | contrast_text     | Test   | Modality   | Significance   |   Sig Max |    Sig Mean |      Sig Std |   % significant voxels |   % left |   % right |
|-----------------------------------+-------------+-------------------+--------+------------+----------------+-----------+-------------+--------------+------------------------+----------+-----------|
| tbss_FA_tfce_corrp_tstat1.nii.gz  | 1. -1.  0.  | Group 1 > Group 2 | t      | FA         | True           |  0.994951 |   0.017583  |   0.00962737 |                   22.4 |     12   |      10.4 |
| tbss_FA_tfce_corrp_tstat2.nii.gz  | -1.  1.  0. | Group 1 < Group 2 | t      | FA         | False          |  0.356494 | nan         | nan          |                  nan   |    nan   |     nan   |
| tbss_FAt_tfce_corrp_tstat1.nii.gz | 1. -1.  0.  | Group 1 > Group 2 | t      | FAt        | True           |  0.991517 |   0.0251597 |   0.0122004  |                    6.1 |      3   |       3   |
| tbss_FAt_tfce_corrp_tstat2.nii.gz | -1.  1.  0. | Group 1 < Group 2 | t      | FAt        | False          |  0.244799 | nan         | nan          |                  nan   |    nan   |     nan   |
| tbss_FW_tfce_corrp_tstat1.nii.gz  | 1. -1.  0.  | Group 1 > Group 2 | t      | FW         | False          |  0.147445 | nan         | nan          |                  nan   |    nan   |     nan   |
| tbss_FW_tfce_corrp_tstat2.nii.gz  | -1.  1.  0. | Group 1 < Group 2 | t      | FW         | True           |  0.977782 |   0.0396292 |   0.00816982 |                    3.9 |      2.2 |       1.7 |
+-----------------------------------+-------------+-------------------+--------+------------+----------------+-----------+-------------+--------------+------------------------+----------+-----------+

```


```sh
randomise_summary.py --figure
```

![figure_fa](test_tbss/stats_real/tbss_FA_tfce_corrp_tstat1.png)
![figure_fat](test_tbss/stats_real/tbss_FAt_tfce_corrp_tstat1.png)
![figure_fw](test_tbss/stats_real/tbss_FW_tfce_corrp_tstat1.png)



```sh
randomise_summary.py --tbss_fill
```

![figure_fa_fill](test_tbss/stats_real/tbss_FA_tfce_corrp_tstat1_filled.png)
![figure_fat_fill](test_tbss/stats_real/tbss_FAt_tfce_corrp_tstat1_filled.png)
![figure_fw_fill](test_tbss/stats_real/tbss_FW_tfce_corrp_tstat1_filled.png)



### Usage



> Simplest use

It automatically finds `design.mat` and `design.con` in the current directory,
along with `*corrp*nii.gz` images when ran without any options.

```sh
cd RANDOMISE/LOCATION/stats
randomise_summary.py
```



> Individual `*corrp*nii.gz`

Also individual `*corrp*nii.gz` images, `design.mat` and `design.con` in 
different location could be specified with options.

```sh
randomise_summary.py -i stats/tbss_corrp_tstat1.nii.gz

# you can also specify design matrices if they have different naming
randomise_summary.py -i stats/tbss_corrp_tstat1.nii.gz \
                     -d stats/this_is_design_file.mat \
                     -c stats/this_is_contrast_file.mat \

```




> Control p-value threshold

The p-value for significance could be altered, if higher threshold is rquired
by adding extra option `-t` or `--threshold`

```sh
randomise_summary.py -t 0.99
```




> Run FSL's atlas query with the significant cluster

FSL's atlas query returns information about the location of the cluster. If
`-a` or `--atlas` option is given, the script will run atlas query on the 
significant cluster and print the summarized output on screen

```sh
randomise_summary.py -a
```



> Extract values for the significant cluster in each subject

It is a common practice to look at the correlation between the values of each
subject in the significant cluster and their clinical scales.  Simply add `--subject_values` option for this.

```sh
randomise_summary.py --subject_values
```

If your randomise directory does not have the `all_*.nii.gz` (4d merged image), 
specify the directory where the 4d merged images are, with `--merged_img_dir`

```sh
randomise_summary.py --subject_values \
                     --merged_img_dir /DIRECTORY/WITH/all_4D_skeleton.nii.gz
```



> Create png file -- **under development : link kcho_figure.py**

```sh
randomise_summary.py --figure
```




> All options

```sh
usage: randomise_summary.py [-h] [--directory DIRECTORY]
                            [--input INPUT [INPUT ...]]
                            [--threshold THRESHOLD] [--contrast CONTRAST]
                            [--matrix MATRIX] [--subject_values]
                            [--merged_img_dir MERGED_IMG_DIR] [--atlasquery]
                            [--figure]

        randomise_summary.py --dir /example/randomise/output/dir/
        

optional arguments:
  -h, --help            show this help message and exit
  --directory DIRECTORY, -d DIRECTORY
                        Specify randomise out dir. This this option is given,
                        design.mat and design.con within the directory are
                        read by default.
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Specify randomise out corrp files. If this option is
                        given, --directory input is ignored
  --threshold THRESHOLD, -t THRESHOLD
                        Threshold for the significance
  --contrast CONTRAST, -c CONTRAST
                        Contrast file used for the randomise.
  --matrix MATRIX, -m MATRIX
                        Matrix file used for the randomise
  --subject_values, -s  Print average in the significant cluster for all
                        subjects
  --merged_img_dir MERGED_IMG_DIR, -p MERGED_IMG_DIR
                        Directory that contains merged files
  --atlasquery, -a      Run atlas query on significant corrp files
  --figure, -f          Create figures

Kevin Cho Thursday, August 22, 2019
```




### Test using `test_randomise_summary.py`

```sh
python test_randomise_summary.py
```

Above script checks for whether 
- the modality from the file name is detected correctly
- the `*corrp*nii.gz` is read properly
- the number of significant voxels are the same as that estimated in FSL
- the mean of significant voxels are almost equal to that estimated in FSL (to 4th decimal point)
- the overlap estimated with the harvard oxford atlas equals that estimated in FSL
