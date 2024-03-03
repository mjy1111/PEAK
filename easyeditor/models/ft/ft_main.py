from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .ft_hparams import FTHyperParams
import torch.nn.functional as F

def apply_ft_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_ft(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: FTHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Define inputs
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]
    
    print(requests, texts)


    #jiequ=1
    postive_list=requests[0]["positive"]
    
    negtive_list=requests[0]["negtive"]



    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights

    # Update loop: intervene at layers simultaneously
    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        loss_meter.reset()

        for txt, tgt in zip(
            chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            inputs = tok(txt, return_tensors="pt", padding=True).to(device)
            target_ids = tok(tgt, return_tensors="pt", padding=True)["input_ids"].to(
                device
            )
            last_token_inds = inputs["attention_mask"].sum(dim=1) - 1
            loss_mask = target_ids != tok.unk_token_id
            opt.zero_grad()
            bs = inputs["input_ids"].shape[0]
            

            #all positive examples
            po_target_ids = []
            for po in postive_list:
                po_target_ids.append(tok(" "+po, return_tensors="pt").to("cuda")[
                    "input_ids"
                ][0])
            #print(po_target_ids)
            
            #choose a negtive example**************************************  lama************
        
            neg_target_ids = []
            for neg in negtive_list:
                neg_target_ids.append(tok(" "+neg, return_tensors="pt").to("cuda")[
                    "input_ids"
                ][0])




            if 't5' in hparams.model_name.lower():
                inputs['labels'] = target_ids
                logits = model(**inputs).logits
                unmasked_log_probs = logits.log_softmax(-1).gather(-1, inputs['labels'].unsqueeze(-1)).squeeze(-1)

                mask = inputs['labels'] != -100
                n_tokens = mask.float().sum()
                avg_log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
                nll = -avg_log_prob
                loss = nll
            else:
                probs = torch.nn.functional.log_softmax(
                    model(**inputs).logits[torch.arange(bs), last_token_inds], dim=-1
                )
                #print(bs,last_token_inds, model(**inputs).logits, '\n\n', model(**inputs).logits[torch.arange(bs), last_token_inds],  torch.arange(bs), probs.shape, target_ids.shape)
                loss_tar = -(torch.gather(probs, 1, target_ids) * loss_mask).sum(
                    1
                ) / loss_mask.sum(1)
                
                #print(torch.gather(probs, 1, target_ids).shape, torch.gather(probs, 1, target_ids))
                
                
                # **********************new added ***********************
                po_target_loss_list = None
                for p in postive_list:
                    pos_target_ids = tok([p], return_tensors="pt", padding=True)["input_ids"].to(device)
                    po_loss_mask = pos_target_ids != tok.unk_token_id
                    if po_target_loss_list == None:
                        po_target_loss_list = -(torch.gather(probs, 1, pos_target_ids) * po_loss_mask).sum(1) / po_loss_mask.sum(1)
                    else:
                        po_target_loss_list = torch.concat((po_target_loss_list,-(torch.gather(probs, 1, pos_target_ids) * po_loss_mask).sum(1) / po_loss_mask.sum(1)),dim=0)
                    
                #po_nll_loss_each = torch.tensor(po_target_loss_list)
                
                po_nll_loss_each = po_target_loss_list
                #print(po_nll_loss_each)
                
                neg_target_loss_list = None
                for n in negtive_list:
                    neg_target_ids = tok([n], return_tensors="pt", padding=True)["input_ids"].to(device)
                    neg_loss_mask = neg_target_ids != tok.unk_token_id
                    #neg_target_loss_list.append(-(torch.gather(probs, 1, neg_target_ids) * neg_loss_mask).sum(1) / neg_loss_mask.sum(1))
                    if neg_target_loss_list == None:
                        neg_target_loss_list = -(torch.gather(probs, 1, neg_target_ids) * neg_loss_mask).sum(1) / neg_loss_mask.sum(1)
                    else:
                        neg_target_loss_list = torch.concat((neg_target_loss_list,-(torch.gather(probs, 1, neg_target_ids) * neg_loss_mask).sum(1) / neg_loss_mask.sum(1)),dim=0)



                #neg_nll_loss_each = torch.tensor(neg_target_loss_list)
                
                neg_nll_loss_each = neg_target_loss_list
                
                if it==0:
                    po_nll_loss_copy = po_nll_loss_each
                    neg_nll_loss_copy = neg_nll_loss_each


                #hinge_previous
                
                po_hinge_loss = F.relu(-po_nll_loss_copy + po_nll_loss_each).mean()
                
                neg_hinge_loss = F.relu(-neg_nll_loss_each + neg_nll_loss_copy).mean()
        
        
                #hinge_loss
                margin = 2
                epsilon = 1e-6
        
                score_orig = torch.concat((-po_nll_loss_each,-neg_nll_loss_each),dim=0)
                N = score_orig.shape[0]
                score_1 = score_orig.expand(N, N).to(f"cuda:{hparams.device}")       # score_1 shape (N, N)
                score_2 = score_1.transpose(1, 0).to(f"cuda:{hparams.device}")
                label = torch.zeros(score_orig.shape)
                label[:po_nll_loss_each.shape[0]] = 1.0
                
                #print(po_nll_loss_each, neg_nll_loss_each, score_orig, score_1, score_2)
                
                
                label_1 = label.expand(N, N).to(f"cuda:{hparams.device}")      # label_1 shape (N, N)
                label_2 = label_1.transpose(0, 1).to(f"cuda:{hparams.device}")
                label_diff = F.relu(label_1 - label_2)
                score_diff = F.relu(score_2 - score_1 + margin)
                hinge_loss = torch.sum(score_diff * label_diff) / (torch.sum(label_diff) + epsilon) # 标准化处理，加上epsilon防止溢出
                #hinge_loss

                loss = loss_tar.mean() + hparams.beta*po_hinge_loss + hparams.gama*neg_hinge_loss +hparams.aerfa*hinge_loss
                #loss.requires_grad_(True)
                
                print(loss_tar, hinge_loss, po_hinge_loss, neg_hinge_loss)
            print(f"Batch loss {loss.item()}")
            loss_meter.update(loss.item(), n=bs)

            if loss.item() >= 1e-2:
                loss.backward(retain_graph=True)
                opt.step()

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )

        print(f"Total loss {loss_meter.avg}")

        if loss_meter.avg < 1e-2:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
