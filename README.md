# PEAK
The repository for our paper: 

Neighboring Perturbations of Knowledge Editing on Large Language Models ([arxiv](https://arxiv.org/abs/2401.17623)).

## 🔔News
- **2024-05-02  Delighted to announce that this paper has been accepted to ICML 2024.**


## Overview
**knowledge editing** aims at efficiently altering LLMs’ behaviors within specific domains while preserving overall performance across various inputs.
Previous primarily focus on determining if the new target knowledge has been successfully memorized. However, the **perturbations** of editing on knowledge neighboring to the new target knowledge have not been fully explored when updating new knowledge to LLMs.

This paper investigates whether the editing operation of appending a new answer into an answer list to a factual question perturbs the neighboring knowledge encapsulated within them.
It also proposes a plug-and-play framework termed APP to mitigate the neighboring perturbation by maintaining the integrity of the answer list.

<img src="https://github.com/mjy1111/PEAK/blob/main/definition.png" width="600">

## Datasets
The PEAK benchmark comprises two datasets of PEAK_counter and PEAK_time, which are included in `data/`.

* `PEAK_counter.json`: the counterfactual dataset for the evaluation of knowledge editing methods on counterfactual appending.
* `PEAK_time.json`: temporal knowledge edits of changes in the real-world. 

The whole data directory is as follows:
```bash
data/
    |__ PEAK_counter.json
    |__ PEAK_time.json
```


## Prepare the environment

### Requirements

**Note: Please use Python 3.9+**
To get started, simply install conda and run:

```shell
git clone https://github.com/mjy1111/PEAK.git
conda create -n PEAK python=3.9.7
...
pip install -r requirements.txt
```

### Models
All models are putted in `hugging_cache/<model_name>` (model_name=gpt2-xl, gpt-j-6B, llama-7b, or llama2-7b).
These could be changed in `hparams/<method_name>/`.


## Evaluation
The performance of knowledge editing is measured from these dimensions:

- `Efficacy`: whether the edited models could recall the exact editing fact under editing prompts
- `Generalization`: whether the edited models could recall the editing fact under paraphrase prompts
- `Locality`: whether the output of the edited models for inputs out of editing scope remains unchanged after editing
- `Additivity`: the degree of perturbation to neighboring knowledge when appending.

GPT-2 XL (1.5B), GPT-J (6B), and LLaMA-2 (7B) are used for editing.

- These model editing methods are used in our paper as follows:
  - [FT](https://github.com/kmeng01/rome): Fine-Tuning with $L_\infty$ constraint
  - [MEND](https://github.com/eric-mitchell/mend): Mitchell et al. Hypernetwork
  - [KN](https://github.com/Hunter-DDM/knowledge-neurons): Damai Dai et al. Locate then Edit
  - [ROME](https://github.com/kmeng01/rome): Kevin Meng et al. Locate and Edit
  - [MEMIT](https://github.com/kmeng01/memit): Kevin Meng et al. Locate and Edit


### Running the evaluation
After downloading the datasets and models, to get started (e.g. using ROME to edit GPT-2 XL on PEAK_counter dataset), run:
```bash
python neighbor.py \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --ds_name=counter (time for PEAK_time dataset) \
    --cuda=0 \
    --dataset_size=100 (optional)
```

If use the proposed APP, run:

```bash
python neighbor.py \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --ds_name=counter \
    --cuda=0 \
    --aerfa=0.2 \
    --beta=0.2 \
    --gama=0.1 \
    --dataset_size=100 (optional)
```
Results from each run are stored at `results/<data_name>/<method_name>/run_<run_id>`.

To summarize the results (e.g. using ROME to edit GPT-2 XL on PEAK_counter dataset), run:

```bash
python -m experiments.summarize  --dir_name=counter/ROME/gpt2-xl
```

All params are in the `hparams/<method_name>/`, and you can change them as needed.

For ROME and MEMIT, we also provide Wikipedia stats [[Google Drive]](https://drive.google.com/file/d/1DrHW5rQ3_0rNHSsH2vFBtv7ePGNHiVj7/view?usp=drive_link).

### MEND
To use the MEND method, you should firstly download weights here. [[Google Drive]](https://drive.google.com/file/d/1o9uJUEXExda5M-kyvvyFZ3yAC9tmW9gx/view?usp=drive_link).
Then use the same steps above to edit models.

## Citation
If you use this code and dataset, please cite our paper:
```bibtex
@misc{ma2024neighboring,
      title={Neighboring Perturbations of Knowledge Editing on Large Language Models}, 
      author={Jun-Yu Ma and Jia-Chen Gu and Ningyu Zhang and Zhen-Hua Ling},
      year={2024},
      eprint={2401.17623},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
### Questions?
If you have any questions related to the repository or the paper, or you encounter any problems when using the datasets/code, feel free to email Junyu Ma `(mjy1999@mail.ustc.edu.cn)` or open an issue!


### Related Projects
- [EasyEdit](https://github.com/zjunlp/EasyEdit)
- [ROME](https://github.com/kmeng01/rome)
- [FastEdit](https://github.com/hiyouga/FastEdit)

We express sincere gratitude to [EasyEdit](https://github.com/zjunlp/EasyEdit) and [ROME](https://github.com/kmeng01/rome), as we have utilized portions of their source code in our project.




