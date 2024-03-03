"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..util.generate import generate_fast
from ..util.perplexity import perplexity


def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record[x] for x in ["subject", "target_new", "ground_truth"]
    )
    rewrite_prompts = [record["prompt"]]
    paraphrase_prompts = record["rephrase_prompts"]
    neighborhood_prompts = record["locality_prompts"]
    #generation_prompts = record["generation_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new,
        target_true,
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret={}
    ret.update({
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } )
    ret.update({
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    })



    return ret


def compute_rewrite_quality_bicounterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new = (
        record[x] for x in ["subject", "target_new"]
    )
    rewrite_prompts = [record["prompt"]]
    paraphrase_prompts = record["rephrase_prompts"]

    positive_list = record["positive"]
    negtive_list = record["negtive"]

    negtive_random_list = record["negtive_random"]

    neighborhoods = record["locality_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]

    # Form a list of lists of prefixes to test.
    neighborhood_prompts = [neighborhood[0] for neighborhood in neighborhoods]
    neighborhood_answers = [neighborhood[1] for neighborhood in neighborhoods]
    
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        target_new,
        positive_list,
        negtive_list,
        negtive_random_list,
    )

    locality_probs, targets_correct = locality_prediction(
        model,
        tok,
        neighborhood_prompts,
        neighborhood_answers,
        target_new,
    )

    print(probs)

    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]

    # for locality prompts
    cutoffs1 = [0] + np.cumsum(list(map(len, neighborhood_prompts))).tolist()
    ret_probs1 = [locality_probs[cutoffs1[i - 1] : cutoffs1[i]] for i in range(1, len(cutoffs1))]
    
    print('\n',111,cutoffs, ret_probs,'\n')
    # Structure the restuls as a dictionary.
    ret={}
    ret.update({
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    } )
    ret.update({
        f"{key}_probs": ret_probs1[i]
        for i, key in enumerate(
            [
                "neighborhood_prompts",
            ]
        )
    } )
    #reverse_judge_compute_probs
    model_name = model.config._name_or_path.replace("/", "_")
    #generation_prompts = record["generation_prompts"]


    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    positive_list,
    negtive_list,
    negtive_random_list,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """
    device=model.device
    print(device)
    
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    
    all_candidates=[target_new]
    all_candidates.extend(positive_list)
    all_candidates.extend(negtive_list)
    all_candidates.extend(negtive_random_list)
    
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in all_candidates
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")


    #print(prefixes)
    #print(tok(" aaa")["input_ids"])
    all_candidates_tok = [tok(f" {n}")["input_ids"] for n in all_candidates]
    all_candidates_tok_len = [len(n) for n in all_candidates_tok]
    

    #model_name = model.config._name_or_path.replace("/", "_")
    if hasattr(model.config,'_name_or_path'):
        model_name = model.config._name_or_path.replace("/", "_")
    else:
        model_name = model.config.model_name
    #print(model_name)
    #print(model.config)
    if 'llama' in model_name.lower() or "vicuna" in model_name.lower():
        #print(1)
        all_candidates_tok = [tok(f"{n}")["input_ids"] for n in all_candidates]
        for i in range(len(all_candidates_tok)):
            all_candidates_tok[i] = all_candidates_tok[i][1:]
        all_candidates_tok_len = [len(n) for n in all_candidates_tok]

    with torch.no_grad():
        if hasattr(model.config,'_name_or_path'):
            logits = model(**prompt_tok).logits
        else:
            logits = model(**prompt_tok)
        #print(model(**prompt_tok))
    '''
    print(prefix_lens)
    print(prompt_tok)
    print(logits.shape)
    print(choice_a_len, choice_b_len)
    '''

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    #print(len(positive_list), len(negtive_list), probs.shape,logits.shape, all_candidates_tok_len)
    targets_correct = []

    for i in range(logits.size(0)):

        cur_len = all_candidates_tok_len[i % len(all_candidates)]
        #choice_a_len if i % len() == 0 else choice_b_len
        '''
        #for lama
        if i % 2 == 0:
            n=target_new
        else:
            n=target_true
        if prefix_lens[i//2]+cur_len !=len(tok(prefixes[i//2]+f" {n}")["input_ids"]):
            print("lama tokenize has a 0")
            #delta=len(tok(prefix_lens[i]+f" {n}")["input_ids"])- len(tok(prefix_lens[i])["input_ids"])
            #cur_len=len(tok(prefixes[i//2]+f" {n}")["input_ids"])-prefix_lens[i//2]
            cur_len=cur_len-2
        '''
        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = all_candidates_tok[i % len(all_candidates)][j]
            #print(cur_tok)

            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // len(all_candidates)] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
    
    #print(probs)
    return [
        {"target_new": probs[i].item(), "positive_list": probs[i+1 : i+1+len(positive_list)].tolist(), "negtive_list": probs[i+1+len(positive_list): i+1+len(positive_list)+len(negtive_list)].tolist(), "negtive_random_list": probs[i+1+len(positive_list)+len(negtive_list): i+1+len(positive_list)+len(negtive_list)+len(negtive_random_list)].tolist()}
        for i in range(0, len(probs), len(all_candidates))
    ], targets_correct


# locality_prediction(
#         model,
#         tok,
#         neighborhood_prompts,
#         neighborhood_answers,
#         target_new,
#     )

def locality_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    neighborhood_answers,
    target_new: str,
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """
    device=model.device
    #print(device)
    
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    
    all_candidates=[target_new]
    all_candidates.extend(neighborhood_answers)
    
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in all_candidates
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")


    #print(prefixes)
    #print(tok(" aaa")["input_ids"])
    all_candidates_tok = [tok(f" {n}")["input_ids"] for n in all_candidates]
    all_candidates_tok_len = [len(n) for n in all_candidates_tok]
    

    #model_name = model.config._name_or_path.replace("/", "_")
    if hasattr(model.config,'_name_or_path'):
        model_name = model.config._name_or_path.replace("/", "_")
    else:
        model_name = model.config.model_name
    #print(model_name)
    #print(model.config)
    if 'llama' in model_name.lower() or "vicuna" in model_name.lower():
        #print(1)
        all_candidates_tok = [tok(f"{n}")["input_ids"] for n in all_candidates]
        for i in range(len(all_candidates_tok)):
            all_candidates_tok[i] = all_candidates_tok[i][1:]
        all_candidates_tok_len = [len(n) for n in all_candidates_tok]

    with torch.no_grad():
        if hasattr(model.config,'_name_or_path'):
            logits = model(**prompt_tok).logits
        else:
            logits = model(**prompt_tok)
        #print(model(**prompt_tok))
    '''
    print(prefix_lens)
    print(prompt_tok)
    print(logits.shape)
    print(choice_a_len, choice_b_len)
    '''

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    #print(len(positive_list), len(negtive_list), probs.shape,logits.shape, all_candidates_tok_len)
    targets_correct = []

    for i in range(logits.size(0)):

        cur_len = all_candidates_tok_len[i % len(all_candidates)]
        #choice_a_len if i % len() == 0 else choice_b_len
        '''
        #for lama
        if i % 2 == 0:
            n=target_new
        else:
            n=target_true
        if prefix_lens[i//2]+cur_len !=len(tok(prefixes[i//2]+f" {n}")["input_ids"]):
            print("lama tokenize has a 0")
            #delta=len(tok(prefix_lens[i]+f" {n}")["input_ids"])- len(tok(prefix_lens[i])["input_ids"])
            #cur_len=len(tok(prefixes[i//2]+f" {n}")["input_ids"])-prefix_lens[i//2]
            cur_len=cur_len-2
        '''
        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = all_candidates_tok[i % len(all_candidates)][j]
            #print(cur_tok)

            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // len(all_candidates)] + j - 1, :], dim=0
            )[cur_tok].item()
        probs[i] /= cur_len

        # Compute accuracy on new targets
    
    #print(probs)
    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1+ i % len(all_candidates)].tolist()}
        for i in range(0, len(probs), len(all_candidates))
    ], targets_correct



def test_seq2seq_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    target_true: str,
):
    """ """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    input_tok = tok(
        [
            f"{prefix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    target_tok = tok(
        [
            f"{suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    prompt_tok = dict()
    prompt_tok.update(input_tok)

    prompt_tok['decoder_input_ids'] = target_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = target_tok['attention_mask']

    a_tok, b_tok = (tok(f"{n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i,  j, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]


# def test_generation(
#     model,
#     tok,
#     prefixes: typing.List[str],
#     consistency_texts: typing.List[str],
#     essence_texts: typing.List[str],
#     # vec: TfidfVectorizer,
# ):
#     gen_texts = generate_fast(
#         model,
#         tok,
#         prefixes,
#         n_gen_per_prompt=1,
#         max_out_len=100,
#     )
#
#     ngram_entropy = n_gram_entropy(gen_texts)
#     consistency_tfidf = tfidf_similarity(
#         " ".join(gen_texts), " ".join(consistency_texts), vec
#     )
#
#     ret = {
#         "ngram_entropy": ngram_entropy,
#         "reference_score": consistency_tfidf,
#         "text": gen_texts,
#     }
#
#     if len(essence_texts) > 0:
#         ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
#         ret.update({"essence_score": ppl, "essence_text": essence_texts})
#
#     return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
