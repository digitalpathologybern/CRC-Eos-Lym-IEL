# Deep learning-based quantification of eosinophils and lymphocytes shows complementary prognostic effects and interplay in patients with colorectal cancer

## Code to generate the results from the published paper:
[Deep learning-based quantification of eosinophils and lymphocytes shows complementary prognostic effects and interplay in patients with colorectal cancer](https://www.nature.com/articles/s41698-025-00955-0)

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
@article{baumann2025deep,
  title={Deep learning-based quantification of eosinophils and lymphocytes shows complementary prognostic effects in colorectal cancer patients},
  author={Baumann, Elias and Lechner, Sophie and Krebs, Philippe and Kirsch, Richard and Berger, Martin D and Lugli, Alessandro and Nagtegaal, Iris D and Perren, Aurel and Zlobec, Inti},
  journal={npj Precision Oncology},
  volume={9},
  number={1},
  pages={1--11},
  year={2025},
  publisher={Nature Publishing Group}
}

```
