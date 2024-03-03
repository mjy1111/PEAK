from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from ..rome import repr_tools
from ...util import nethook

from .rome_hparams import ROMEHyperParams
from .const import relationid_words


def locate_tokenize(model_name, tok, requests_reverse,subject):

    #print(requests_reverse, subject)
    if 'llama' in model_name.lower() or "vicuna" in model_name.lower():
        
        if len(tok.tokenize(requests_reverse))>len(tok.tokenize(subject)):
            sub_tokenize=[len(tok.tokenize(requests_reverse))-len(tok.tokenize(subject)) +1 , len(tok.tokenize(requests_reverse)) + 1]
        else:
            sub_tokenize=[1, len(tok.tokenize(subject))+1]
    else:
        if len(tok.tokenize(requests_reverse))>len(tok.tokenize(subject)):
            sub_tokenize=[len(tok.tokenize(requests_reverse))-len(tok.tokenize(subject)), len(tok.tokenize(requests_reverse))]
        else:
            sub_tokenize=[0,len(tok.tokenize(subject))]
    return {"s":sub_tokenize}


def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"], return_tensors="pt").to(f"cuda:{hparams.device}")[
        "input_ids"
    ][0]

    model_name = model.config._name_or_path.replace("/", "_")
    model_layers=model.config.n_layer if "gpt" in model_name else model.config.num_hidden_layers


    #print(model.config._name_or_path)
    if 'llama' in model_name.lower() or "vicuna" in model_name.lower():
        #print(12)
        target_ids = target_ids[1:]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context in context_templates
    ], ["{} is a"]
    
    
    
    target_new=request["target_new"][1:]
    print(request)
    
    
    
    #choose postive*********************************
    #postive_list=request["positive"]
    postive_list=request["positive"]
    
    negtive_list=request["negtive"]

    
    #all positive examples
    po_target_ids = []
    for po in postive_list:
        if 'llama' in model_name.lower() or "vicuna" in model_name.lower():
            po_target_ids.append(tok(" "+po, return_tensors="pt").to("cuda")[
                "input_ids"
            ][0][1:])
        else:
            po_target_ids.append(tok(" "+po, return_tensors="pt").to("cuda")[
                "input_ids"
            ][0])
    #print(po_target_ids)
    
    po_rewriting_prompts = [
        request["prompt"] + tok.decode(po_target_id[:-1])
        for po_target_id in po_target_ids
    ] 
    print("\n\n po_rewriting_prompts \n\n", po_rewriting_prompts, len(po_rewriting_prompts))
    
    #choose a negtive example**************************************  lama************

    neg_target_ids = []
    for neg in negtive_list:
        if 'llama' in model_name.lower() or "vicuna" in model_name.lower():
            neg_target_ids.append(tok(" "+neg, return_tensors="pt").to("cuda")[
                "input_ids"
            ][0][1:])
        else:
            neg_target_ids.append(tok(" "+neg, return_tensors="pt").to("cuda")[
                "input_ids"
            ][0])
    #print(po_target_ids)
    
    neg_rewriting_prompts = [
        request["prompt"] + tok.decode(neg_target_id[:-1])
        for neg_target_id in neg_target_ids
    ]


    print("\n\n neg_rewriting_prompts \n\n", neg_rewriting_prompts, len(neg_rewriting_prompts))

    # hrt_prompts = context_templates
    # hrt_obj_prompts = [
    #     context.format(target_new)
    #     for context in context_templates
    # ]

    '''
    relation_word=relationid_words[request["relation_id"]][0]
    relation_word_inv=relationid_words[request["relation_id"]][1]
    #print(relation_word,request["ground_truth"])
    '''


    location=[]
    location_re=[]
    location_re_inv=[]
    location_ob=[]
    location_ob_true=[]

    all_prompts = rewriting_prompts + kl_prompts + po_rewriting_prompts + neg_rewriting_prompts

    print(target_ids[:-1],  tok.decode(target_ids[:-1]), all_prompts,len(all_prompts),'\n')

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    #*********format(subject)**********
    print(lookup_idxs)

    # for i in range(len(hrt_prompts)):
    #     all_prompts[len(rewriting_prompts+kl_prompts)+i] = all_prompts[len(rewriting_prompts+kl_prompts)+i].format(request["subject"])
    
    print(all_prompts, len(all_prompts))


    #kl prompt 在里面
    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")



    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    
    print('\n\n',len(rewriting_prompts), rewriting_targets, rewriting_targets.shape, *input_tok["input_ids"].shape[1:],'\n\n')
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
    
    #print(rewriting_targets)
    

    # Compute positive  rewriting targets*****************************
    po_rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(po_rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )

    for i in range(len(po_rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i + len(rewriting_prompts + kl_prompts)].sum()
        po_rewriting_targets[i, ex_len - len(po_target_ids[i]) : ex_len] = po_target_ids[i]    
    
    
    # Compute negtive  rewriting targets***********************
    neg_rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(neg_rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )

    for i in range(len(neg_rewriting_targets)):
        ex_len = input_tok["attention_mask"][i + len(rewriting_prompts + kl_prompts + po_rewriting_prompts)].sum()
        neg_rewriting_targets[i, ex_len - len(neg_target_ids[i]) : ex_len] = neg_target_ids[i] 
    

    # Compute indices of the tokens where the fact is looked up


    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            # logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(po_rewriting_prompts + neg_rewriting_prompts + kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts)-len(po_rewriting_prompts + neg_rewriting_prompts) :-len(po_rewriting_prompts + neg_rewriting_prompts)])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()


        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)
        

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        #print(loss.shape, loss)
        mask = (rewriting_targets != -100).float()
        
        
        #print(torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).shape)
        

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        #print('\n\n', loss, loss * mask, loss.shape, nll_loss_each.shape, nll_loss.shape, target_ids.shape, '\n\n')


        #positve nll loss*******************************
        po_loss = torch.gather(
            log_probs[len(rewriting_prompts + kl_prompts):],
            2,
            torch.where(po_rewriting_targets != -100, po_rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        #print(loss.shape, loss)
        mask = (po_rewriting_targets != -100).float()
        
        #print(po_rewriting_targets)
        
        # Aggregate total losses
        po_nll_loss_each = -(po_loss * mask).sum(1)
        
        len_mask = mask.sum(1)
        
        #print(po_nll_loss_each, len_mask)
        po_nll_loss_each = po_nll_loss_each / len_mask
        
        #print(po_nll_loss_each)
        
        po_nll_loss = po_nll_loss_each.mean()
        
        print(po_nll_loss_each.shape, po_nll_loss.shape)


        #negtive nll loss***********************************
        neg_loss = torch.gather(
            log_probs[len(rewriting_prompts + kl_prompts+ po_rewriting_prompts):],
            2,
            torch.where(neg_rewriting_targets != -100, neg_rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        #print(loss.shape, loss)
        mask = (neg_rewriting_targets != -100).float()
        
        #print(torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).shape)
        # Aggregate total losses
        neg_nll_loss_each = -(neg_loss * mask).sum(1)
        
        len_mask = mask.sum(1)
        neg_nll_loss_each = neg_nll_loss_each / len_mask
        
        neg_nll_loss = neg_nll_loss_each.mean()

        if it==0:
            neg_nll_loss_copy = neg_nll_loss_each.clone().detach()
            po_nll_loss_copy = po_nll_loss_each.clone().detach()
        
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
        
        #print(score_diff,'\n', label_diff,"\n")



        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        #loss = nll_loss + po_nll_loss - neg_nll_loss + kl_loss + weight_decay + hparams.aerfa*hrt_loss
        loss= nll_loss + hparams.beta*po_hinge_loss + hparams.gama*neg_hinge_loss +hparams.aerfa*hinge_loss  + kl_loss + weight_decay


        #loss = nll_loss + hparams.gama*hinge_loss + hparams.aerfa * po_nll_loss - hparams.beta * neg_nll_loss + kl_loss + weight_decay

        #loss = nll_loss + hparams.aerfa*hinge_loss + kl_loss + weight_decay
        print(nll_loss, hinge_loss, po_hinge_loss, neg_hinge_loss)
        #loss = 0.1*nll_loss + kl_loss + weight_decay + hparams.aerfa*hrt_loss
        #print(po_nll_loss, kl_loss, weight_decay)

        #print(hparams.gama, hparams.aerfa, hparams.beta, hparams.v_num_grad_steps)

        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(hparams.gama*nll_loss.item(), 3)} + {np.round(po_nll_loss.item(), 3)} + {np.round(neg_nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)}"
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
            f"avg prob of [{postive_list[0]}] "
            f"{torch.exp(-po_nll_loss_each).mean().item()}"
            f"avg prob of [{negtive_list[0]}] "
            f"{torch.exp(-neg_nll_loss_each).mean().item()}"
        )
        #print(hparams.aerfa)


        if nll_loss < 5e-2:
            break


        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
