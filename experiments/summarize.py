import collections
import json
from pprint import pprint
from typing import List, Optional

import numpy as np
from scipy.stats import hmean
from pathlib import Path
#from util.globals import *

import math
 
 
def sigmoid_function(num):

    return  1/(1 + math.exp(-num))





def rank_compute_affect(positive_list, negtive_list):

    exp_negtive_list=[]
    for x in negtive_list:
        exp_negtive_list.append(sigmoid_function(np.exp(-x)))
    
    exp_positive_list=[]
    for x in positive_list:
        exp_positive_list.append(sigmoid_function(np.exp(-x)))


    max_neg=max(exp_negtive_list)
    min_pos=min(exp_positive_list)

    neg_dayu=0
    pos_xiaoyu=0
    
    neg_sum=0
    pos_sum=0

    for i in exp_negtive_list:
        neg_sum+=i
    for i in exp_positive_list:
        pos_sum+=i  
    
    
    for i in exp_negtive_list:
        if i > min_pos:
            neg_dayu += i
       

    for i in exp_positive_list:
        if i < max_neg:
            pos_xiaoyu += i

    rate_neg2pos=float(neg_dayu/neg_sum)
    rate_pos2neg = float(pos_xiaoyu /pos_sum)
    #print(rate_neg2pos, rate_pos2neg)
    return rate_neg2pos, rate_pos2neg

def rank_compute_affect_post(positive_list, negtive_list, positive_list_pre, negtive_list_pre):

    exp_negtive_list=[]
    for x in negtive_list:
        exp_negtive_list.append(sigmoid_function(np.exp(-x)))
    
    exp_positive_list=[]
    for x in positive_list:
        exp_positive_list.append(sigmoid_function(np.exp(-x)))
    '''
    for x in negtive_list:
        exp_negtive_list.append(np.exp(-x))
    
    exp_positive_list=[]
    for x in positive_list:
        exp_positive_list.append(np.exp(-x))
    '''
    #use for whole probability
    exp_negtive_list_post=[]
    for x in negtive_list:
        exp_negtive_list_post.append(np.exp(-x))
    
    exp_positive_list_post=[]
    for x in positive_list:
        exp_positive_list_post.append(np.exp(-x))

    exp_negtive_list_pre=[]
    for x in negtive_list_pre:
        exp_negtive_list_pre.append(np.exp(-x))
    
    exp_positive_list_pre=[]
    for x in positive_list_pre:
        exp_positive_list_pre.append(np.exp(-x))
    
    mean_neg_pre = np.mean(exp_negtive_list_pre)
    mean_pos_pre = np.mean(exp_positive_list_pre)
    
    #post
    mean_neg_post = np.mean(exp_negtive_list_post)
    mean_pos_post = np.mean(exp_positive_list_post)

    max_neg=max(exp_negtive_list)
    min_pos=min(exp_positive_list)

    neg_dayu=0
    pos_xiaoyu=0
    
    neg_sum=0
    pos_sum=0

    for i in exp_negtive_list:
        neg_sum+=i
    for i in exp_positive_list:
        pos_sum+=i  
    
    
    for i in exp_negtive_list:
        if i > min_pos:
            neg_dayu += i
       

    for i in exp_positive_list:
        if i < max_neg:
            pos_xiaoyu += i

    rate_neg2pos=float(neg_dayu/neg_sum)
    rate_pos2neg = float(pos_xiaoyu /pos_sum)

    #rate_neg2pos = 1-(1-rate_neg2pos) * mean_neg_post/
    if mean_pos_post/mean_pos_pre < 1:
        rate_pos2neg = 1-(1-rate_pos2neg) * mean_pos_post/mean_pos_pre

    if mean_neg_post/mean_neg_pre > 1:
        rate_neg2pos = 1-(1-rate_neg2pos) * mean_neg_pre/mean_neg_post

    #rate_pos2neg = max(1, 2/(1 + math.exp(-(mean_pos_post/mean_pos_pre-1))))*rate_pos2neg

    #print(mean_pos_post/mean_pos_pre
    #print(rate_neg2pos, rate_pos2neg)
    return rate_neg2pos, rate_pos2neg




def select_pos_neg(positive_list, negtive_list, negtive_random_list):




    exp_negtive_list=[]
    for x in negtive_list:
        exp_negtive_list.append(np.exp(-x))
    
    exp_positive_list=[]
    for x in positive_list:
        exp_positive_list.append(np.exp(-x))


    exp_negtive_random_list=[]
    for x in negtive_random_list:
        exp_negtive_random_list.append(np.exp(-x))



    exp_negtive_list_copy=exp_negtive_list.copy()
    exp_positive_list_copy=exp_positive_list.copy()
    exp_negtive_random_list_copy = exp_negtive_random_list.copy()
    
    '''
    for i in exp_negtive_list:
        baifen=0
        for j in exp_positive_list_copy:
            if i > j:
                baifen+=1
        if float(baifen) / len(exp_positive_list) >= 0.25:
            exp_negtive_list_copy.remove(i)
    '''

    for i in exp_positive_list:
        baifen=0
        for j in exp_negtive_list_copy:
            if i < j:
                baifen+=1
        if float(baifen) / len(exp_negtive_list) >= 0.2:
            exp_positive_list_copy.remove(i)

    #print(exp_negtive_list_copy, exp_positive_list_copy)
    
    
    selected_pos = exp_positive_list_copy.copy()
    selected_neg = exp_negtive_list_copy.copy()

    selected_neg_random = exp_negtive_random_list_copy.copy()
    
    if len(selected_neg)==0 or len(selected_pos)==0 or len(selected_neg_random)==0:
        return [], [], []
    

    for i in selected_neg:
        if i >= min(selected_pos):
            exp_negtive_list_copy.remove(i)

    for i in selected_neg_random:
        if i >= min(selected_pos):
            exp_negtive_random_list_copy.remove(i)

    selected_pos = exp_positive_list_copy.copy()
    selected_neg = exp_negtive_list_copy.copy()
    selected_neg_random = exp_negtive_random_list_copy.copy()

    #print(selected_pos, '\n', selected_neg)
    pos_id=[]
    neg_id=[]
    neg_random_id=[]
    for i in selected_pos:
        pos_id.append(exp_positive_list.index(i))
    for i in selected_neg:
        neg_id.append(exp_negtive_list.index(i))

    for i in selected_neg_random:
        neg_random_id.append(exp_negtive_random_list.index(i))
    #print("pos:", pos_id, '\n',"neg:", neg_id)
    return pos_id, neg_id, neg_random_id



def main(
    dir_name,
    runs: Optional[List],
    first_n_cases=None,
    get_uncompressed=False,
    abs_path=False,
):  # runs = None -> all runs
    summaries = []
    uncompressed = []

    for run_dir in (Path("results/{}".format(dir_name)) if not abs_path else dir_name).iterdir():
        # Skip if we're not interested
        if runs is not None and all(run not in str(run_dir) for run in runs):
            continue

        # Iterate through all case files
        cur_sum = collections.defaultdict(lambda: [])
        files = list(run_dir.glob("*case_*.json"))
        files.sort(key=lambda x: int(str(x).split("_")[-1].split(".")[0]))
        #print(files)
        #print(len(files))
        for case_file in files[:]:
            try:
                with open(case_file, "r") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not decode {case_file} due to format error; skipping.")
            

            case_id = data["case_id"]
            if first_n_cases is not None and case_id >= first_n_cases:
                break

            if "time" in data:
                cur_sum["time"].append(data["time"])

            data["pre"]["rp_prompts_probs"] = []
            data["pre"]["rp_prompts_probs"].extend(data["pre"]["rewrite_prompts_probs"])
            data["pre"]["rp_prompts_probs"].extend(data["pre"]["paraphrase_prompts_probs"])
            data["post"]["rp_prompts_probs"] = []
            data["post"]["rp_prompts_probs"].extend(data["post"]["rewrite_prompts_probs"])
            data["post"]["rp_prompts_probs"].extend(data["post"]["paraphrase_prompts_probs"])


            for prefix in ["pre","post"]:
                # Probability metrics for which new should be lower (better) than true

                for key in ["rp_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue
                    

                    sum_key_neg2pos = f"{prefix}_{key.split('_prompts')[0]}_neg2pos_rate"
                    sum_key_pos2neg = f"{prefix}_{key.split('_prompts')[0]}_pos2neg_rate"

                    sum_key_neg2pos_random = f"{prefix}_{key.split('_prompts')[0]}_random_neg2pos_rate"
                    sum_key_pos2neg_random = f"{prefix}_{key.split('_prompts')[0]}_random_pos2neg_rate"
                    #sum_key_cont = f"{prefix}_{key.split('_prompts')[0]}_diff"
                    
                    if data[prefix][key]==[]:
                        continue
                    
                    neg2pos_allpara=[]
                    pos2neg_allpara=[]

                    neg2pos_random=[]
                    pos2neg_random=[]
                    
                    for index in range(len(data[prefix][key])):
                        
                        x = data[prefix][key][index].copy()
                        
                        pre_x = data["pre"][key][index].copy()
                        if prefix=="pre":
                            pos_id, neg_id, neg_random_id = select_pos_neg(x["positive_list"], x["negtive_list"], x["negtive_random_list"])
                        if prefix=="post":
                            pre_x = data["pre"][key][index]
                            pos_id, neg_id, neg_random_id = select_pos_neg(pre_x["positive_list"], pre_x["negtive_list"], pre_x["negtive_random_list"])
                        #print(pos_id,neg_id)
                            
                        #pos_id, neg_id = select_pos_neg(x["positive_list"], x["negtive_list"])

                        temp_pos = [x["positive_list"][i] for i in pos_id]
                        temp_neg = [x["negtive_list"][i] for i in neg_id]
                        temp_neg_random = [x["negtive_random_list"][i] for i in neg_random_id]
                        
                        temp_pos_pre = [pre_x["positive_list"][i] for i in pos_id]
                        temp_neg_pre = [pre_x["negtive_list"][i] for i in neg_id]
                        temp_neg_random_pre = [pre_x["negtive_random_list"][i] for i in neg_random_id]
                        if len(temp_pos)==0 or len(temp_neg)==0 or len(temp_neg_random)==0:
                            continue
                            
                        if prefix=="pre":
                            p,q=rank_compute_affect(temp_pos, temp_neg)

                            pr,qr=rank_compute_affect(temp_pos, temp_neg_random)
                        
                        #******************use both probbility & rank*************
                        if prefix=="post":
                            #p,q=rank_compute_affect(temp_pos, temp_neg)
                            p,q=rank_compute_affect_post(temp_pos, temp_neg, temp_pos_pre, temp_neg_pre)

                            pr,qr=rank_compute_affect_post(temp_pos, temp_neg_random, temp_pos_pre, temp_neg_random_pre)
                        
                        neg2pos_allpara.append(p)
                        pos2neg_allpara.append(q)

                        neg2pos_random.append(pr)
                        pos2neg_random.append(qr)
                    #print(neg2pos_allpara, pos2neg_allpara) 
                    
                    if len(temp_pos)==0 or len(temp_neg)==0 or len(temp_neg_random)==0:
                        continue
                    cur_sum[sum_key_neg2pos].append(np.mean(neg2pos_allpara))
                    cur_sum[sum_key_pos2neg].append(np.mean(pos2neg_allpara))

                    cur_sum[sum_key_neg2pos_random].append(np.mean(neg2pos_random))
                    cur_sum[sum_key_pos2neg_random].append(np.mean(pos2neg_random))

                for key in ["rewrite_prompts_probs","paraphrase_prompts_probs","rp_prompts_probs"]:
                    if prefix not in data or key not in data[prefix]:
                        continue
                    

                    sum_key_discrete = f"{prefix}_{key.split('_prompts')[0]}_success"
                    sum_key_cont = f"{prefix}_{key.split('_prompts')[0]}_diff"


                    if data[prefix][key]==[]:
                        #print(1)
                        continue
                    #filter
                    '''
                    y=data["pre"][key][0]
                    if y["target_true"] > y["target_new"]:
                        continue
                    '''

                    #pos_id, neg_id = select_pos_neg(x["positive_list"], x["negtive_list"])
                    
                    mm=[]

                    for index in range(len(data[prefix][key])):
                        
                        y= data[prefix][key][index].copy()
                        x=dict()
                        x["target_new"] = y["target_new"]
                        pre_x = data["pre"][key][index].copy()

                        #pos_id, neg_id = select_pos_neg(pre_x["positive_list"], pre_x["negtive_list"])
                        pos_id, neg_id, neg_random_id = select_pos_neg(pre_x["positive_list"], pre_x["negtive_list"], pre_x["negtive_random_list"])
                        if pos_id==[] or neg_id==[] or neg_random_id==[]:
                            continue
                        x["positive_list"] = [y["positive_list"][i] for i in pos_id]
                        x["negtive_list"] = [y["negtive_list"][i] for i in neg_id]
                        x["negtive_random_list"] = [y["negtive_random_list"][i] for i in neg_random_id]

                        #print(x)
                        mm.append(x.copy())
                    if mm == []:
                        continue
                    #print(mm)

                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                max(x["positive_list"]) > x["target_new"] for x in mm
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_new"]) - np.exp(-max(x["positive_list"])) for x in mm
                            ]
                        )
                    )
                    

                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    
                    y= data[prefix][key].copy()
                    local_id = []
                    for l in range(len(data["pre"][key])):
                        if data["pre"][key][l]["target_true"] < data["pre"][key][l]["target_new"]:
                            local_id.append(l)
                    
                    filtered_data = [data[prefix][key][i] for i in local_id]
                    if filtered_data==[]:
                        continue
                    
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in filtered_data
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in filtered_data
                            ]
                        )
                    )




                    #print(cur_sum)
                '''
                # Probability metrics for which true should be lower (better) than new
                sum_key_discrete = f"{prefix}_neighborhood_success"
                sum_key_cont = f"{prefix}_neighborhood_diff"
                key = "neighborhood_prompts_probs"
                if prefix in data and key in data[prefix]:
                    cur_sum[sum_key_discrete].append(
                        np.mean(
                            [
                                x["target_true"] < x["target_new"]
                                for x in data[prefix][key]
                            ]
                        )
                    )
                    cur_sum[sum_key_cont].append(
                        np.mean(
                            [
                                np.exp(-x["target_true"]) - np.exp(-x["target_new"])
                                for x in data[prefix][key]
                            ]
                        )
                    )
                '''

        if len(cur_sum) == 0:
            continue
        


        num_items = len(cur_sum[next(iter(cur_sum.keys()))])
        metadata = {
            "run_dir": str(run_dir),
            "num_cases": num_items,
        }

        uncompressed.append(dict(cur_sum, **metadata))
        

        cur_sum = {k: (np.mean(v), np.std(v)) for k, v in cur_sum.items()}
        
        for k, v in cur_sum.items():
            if all(exclude not in k for exclude in ["essence_score", "time"]):
                # Constant multiplication scales linearly with mean and stddev
                cur_sum[k] = tuple(np.around(z * 100, 2) for z in v)
        '''
        for prefix in ["pre", "post"]:
            for k_efficacy, k_generalization in [
                (
                    f"{prefix}_rewrite_success",
                    f"{prefix}_paraphrase_success",
                ),
                # (
                #     f"{prefix}_rewrite_acc",
                #     f"{prefix}_paraphrase_acc",
                #     f"{prefix}_neighborhood_acc",
                # ),
            ]:
                if all(k in cur_sum for k in [k_efficacy, k_generalization]):
                    hmean_list = [
                        cur_sum[k_efficacy][0],
                        cur_sum[k_generalization][0],
                    ]

                    # if f"{prefix}_ngram_entropy" in cur_sum:
                    #     hmean_list.append(2 ** (cur_sum[f"{prefix}_ngram_entropy"][0] / 100))
                    # if f"{prefix}_reference_score" in cur_sum:
                    #     hmean_list.append(cur_sum[f"{prefix}_reference_score"][0])

                    cur_sum[f"{prefix}_score"] = (hmean(hmean_list), np.nan)
                    break
        '''

        cur_sum.update(metadata)
        pprint(cur_sum)
        summaries.append(cur_sum)

    return uncompressed if get_uncompressed else summaries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir_name", type=str, help="Name of directory to scan for runs."
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="By default, summarizes each run in <dir_name>. "
        "If runs are specified, only evaluates those specific runs.",
    )
    parser.add_argument(
        "--first_n_cases",
        type=int,
        default=None,
        help="Restricts evaluation to first n cases in dataset. "
        "Useful for comparing different in-progress runs on the same slice of data.",
    )
    args = parser.parse_args()

    main(
        args.dir_name,
        None if args.runs is None else args.runs.split(","),
        args.first_n_cases,
    )
