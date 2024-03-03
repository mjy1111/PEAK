
import os,json
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from easyeditor.editors.add_editor import BiEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams

import os,json
from pathlib import Path
from dsets import CounterFactDataset, MultiCounterFactDataset, BiCounterFactDataset
from typing import Tuple, Union
from time import time

import torch

'''
from dsets import (
    CounterFactDataset,
    MultiCounterFactDataset,
)
'''

'''
from easyeditor.dataset import (
    CounterFactDataset,
    MultiCounterFactDataset,
)

from eval_bi.eval_utils_counterfact import compute_rewrite_quality_counterfact
'''
DS_DICT = {
    "mcf": (MultiCounterFactDataset),
    "cf": (CounterFactDataset),
    "counter": (BiCounterFactDataset),
    "time": (BiCounterFactDataset),
}


'''
prompts = ['The mother tongue of China is',
                'What role does Denny Herzig play in football?',
                'What city did Marl Young live when he died?']
ground_truth = ['Eliel Saarinen', 'defender', 'Los Angeles']
target_new = ['Alfred Lahti', 'winger', 'New Orleans']
subject = ['China', 'Denny Herzig', 'Marl Young']
hparams=MEMITHyperParams.from_hparams('hparams/MEMIT/gpt2-xl.yaml')
editor=BaseEditor.from_hparams(hparams)
tok=editor.tok
print(tok.tokenize(" 11sjhwe"))
metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    keep_original_weight=False
)
print(metrics)

'''

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]
#print(type(edited_model))

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    ds_name: str,
    dataset_size_limit: int,
    dir_name: str,
    num_edits: int,
    cuda: int,
    gama: float,
    aerfa: float,
    beta: float,
    v_num_grad_steps: int):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    

    RESULTS_DIR="results/{}".format(ds_name)
    
    if ds_name=="counter":
        DATA_DIR="data/PEAK_counter.json"
    elif ds_name=="time":
        DATA_DIR="data/PEAK_time.json"
    
    continue_from_run=None
    if gama!=0 or aerfa!=0 or beta!=0:
        dir_name=dir_name + '_' + str(aerfa)+'_'+str(beta)+'_'+str(gama)
    '''
    #*****************dir name***************
    if alg_name=="BIRD" and aerfa!=0:
        dir_name=dir_name+'_'+str(aerfa)
    '''
    if continue_from_run is None:
        alg_dir = Path("{}/{}/{}/".format(RESULTS_DIR, dir_name, model_name))
        print(alg_dir)
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = Path("{}/{}/{}/".format(RESULTS_DIR,dir_name,model_name) + f"run_{str(run_id).zfill(3)}")
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")
    


    ds_class = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, size=dataset_size_limit)

    # Iterate through dataset
    
    for record_chunks in chunks(ds, num_edits):
        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]

        #etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

    start = time()
    

    prompts=[record['requested_rewrite']["prompt"].format(record['requested_rewrite']["subject"]) for record in ds]
    ground_truth = [record['requested_rewrite']['target_true']["str"] for record in ds]
    target_new = [record['requested_rewrite']['target_new']["str"] for record in ds]
    #target_new = ground_truth
    subject = [record['requested_rewrite']["subject"] for record in ds]
    relation = [record['requested_rewrite']["relation_id"] for record in ds]
    
    positive_list = [record["postive_list"] for record in ds]
    negtive_list = [record["negtive_list"] for record in ds]
    negtive_random_list = [record["negtive_random_list"] for record in ds]
    
    locality_inputs = [record["neighborhood_prompts"] for record in ds]
    #print(positive_list)

    para=[record['para_add_prompts'] for record in ds]


    if alg_name=="MEMIT":
        hparams=MEMITHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
        hparams.gama=gama
        hparams.aerfa=aerfa
        hparams.beta=beta
    elif alg_name=="ROME":
        hparams=ROMEHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
        hparams.gama=gama
        hparams.aerfa=aerfa
        hparams.beta=beta
        hparams.v_num_grad_steps=v_num_grad_steps
        print(hparams.gama, hparams.aerfa, hparams.beta)
    elif alg_name=="MEND":
        hparams=MENDHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
        hparams.aerfa=aerfa
    elif alg_name=="KN":
        hparams=KNHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
    elif alg_name=="FT":
        hparams=FTHyperParams.from_hparams('hparams/{}/{}.yaml'.format(alg_name, model_name))
        hparams.gama=gama
        hparams.aerfa=aerfa
        hparams.beta=beta

    editor=BiEditor.from_hparams(hparams)
    tok=editor.tok
    #print(1)
    all_metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        positive=positive_list,
        negtive=negtive_list,
        negtive_random = negtive_random_list,
        target_new=target_new,
        subject=subject,
        rephrase_prompts=para,
        keep_original_weight=True,
        relation_id=relation,
        case_result_template=case_result_template,
        num_edits1=num_edits,
        locality_inputs = locality_inputs,
    )
    exec_time = time() - start
    print("Execution took", exec_time)

    # Evaluate new model
    start = time()
    #gen_test_vars = [None, None]
    '''
    i=0
    for record in ds:
        out_file = Path(case_result_template.format(num_edits, record["case_id"]))
        if out_file.exists():
            print(f"Skipping {out_file}; already exists")
            continue

        metrics = {
            "case_id": record["case_id"],
            "grouped_case_ids": case_ids,
            "num_edits": num_edits,
            "requested_rewrite": record["requested_rewrite"],
            "time": exec_time,
            "post": all_metrics[i]["post"],
            "pre": all_metrics[i]["pre"],
        }

        # Dump metrics in .json
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=1)
        i+=1
    '''


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "FT", "KN", "MEND","MEMIT"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=["gpt2-xl", "gpt-j-6B","llama2-7b"],
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["counter","time"],
        default="cf",
        help="Dataset to perform evaluations on. Either CounterFact (cf) or zsRE (zsre).",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--dir_name",
        default="cf",
        help="the directory to save results",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="cuda name",
    )
    parser.add_argument(
        "--v_num_grad_steps",
        type=int,
        default=20,
        help="cuda name",
    )
    parser.add_argument(
        "--gama",
        type=float,
        default=0,
        help="cuda name",
    )
    parser.add_argument(
        "--aerfa",
        type=float,
        default=0,
        help="cuda name",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0,
        help="cuda name",
    )
    args = parser.parse_args()
    if args.dir_name=="cf":
        args.dir_name=args.alg_name

    main(args.alg_name, args.model_name, args.ds_name, args.dataset_size_limit, args.dir_name, args.num_edits, args.cuda, args.gama, args.aerfa,args.beta, args.v_num_grad_steps)