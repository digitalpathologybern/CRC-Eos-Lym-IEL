# Deep learning-based quantification of eosinophils and lymphocytes shows complementary prognostic effects and interplay in patients with colorectal cancer

## Code to generate the results from the published paper:
[Deep learning-based quantification of eosinophils and lymphocytes shows complementary prognostic effects and interplay in patients with colorectal cancer]()

## How to use:
```
calculate_scores.py \
    --wsi_path "list_of_cases.txt" \
    --nuclei_results "/path-to/hover_next_results/cohort/" \
	--segmentation_results "/path-to/srma_results/wsi" \
    --nprocs 16 \
    --output_path "/path-to/output/cohort.csv" \
	--array_tasks 8 \
	--array_id $SLURM_ARRAY_TASK_ID
```

The output file will be one or multiple CSV files with a row for each WSI containing scores for each cell type in tumor front and center (if estimation is available and worked correctly).

This work heavily relies on [HoVer-NeXt](https://github.com/digitalpathologybern/hover_next_inference) and the C2R adaptation (unpublished) of [SRMA](https://github.com/christianabbet/SRA).

## Citation
If you are relying on results from the paper or use the code to generate similar results, please consider citing the our work:
```
TODO
```
