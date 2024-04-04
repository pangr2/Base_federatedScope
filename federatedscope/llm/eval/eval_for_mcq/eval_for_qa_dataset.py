# ref: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import os
import torch
import numpy as np
import pandas as pd
from federatedscope.llm.eval.eval_for_mmlu.categories import \
    subcategories, categories
import json
import transformers

from federatedscope.core.configs.config import global_cfg
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.llm.misc.fschat import FSChatBot
from federatedscope.core.data.utils import download_url
import tarfile

# using given json dataset
transformers.logging.set_verbosity(40)

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    ll = subject.split("_")
    s = ""
    for entry in ll:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(model, tokenizer, dev_df, test_df, device, fschatbot):
    cors = []
    all_probs = []
    # answers = choices[:test_df.shape[1] - 2]

    for i in range(len(test_df)):
        input_ids = test_df[i]['input_ids'][:-2]
        input_prompt = fschatbot.tokenizer.decode(input_ids)
        example = fschatbot.tokenizer.decode(dev_df[1]['input_ids'][:-1]) + '\n'

        input_prompt = example + input_prompt
        # print(input_prompt)
        input_question = fschatbot.tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
        logits = model(input_ids=input_question).logits[0, -1]

        # generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        # model_completion = fschatbot.generate(prompt, generate_kwargs)

        probs = (torch.nn.functional.softmax(
            torch.tensor([
                logits[tokenizer("A").input_ids[-1]],
                logits[tokenizer("B").input_ids[-1]],
                logits[tokenizer("C").input_ids[-1]],
                logits[tokenizer("D").input_ids[-1]],
            ]).float(),
            dim=0,
        ).detach().cpu().numpy())
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        label = fschatbot.tokenizer.decode(test_df[i]['input_ids'][-2])
        print("pred: ", pred, "label: ", label, "probs: ", probs)
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    cors = np.array(cors)
    acc = np.count_nonzero(cors) / np.size(cors)
    all_probs = np.array(all_probs)

    return cors, acc, all_probs


def main(data_set_list, check_point_list):
    results = {"accuracy": [], "data_set": [], "check_point": [], "accuracy_list": []}
    eval_dir = "result_ckpt/eval_result"

    for data_set, check_point in zip(data_set_list, check_point_list):
        print(f'using {data_set} with {check_point}')
        with open(data_set, "r") as f:
            data = json.load(f)

            init_cfg = global_cfg.clone()
            args = parse_args()

            if args.cfg_file:
                init_cfg.merge_from_file(args.cfg_file)
            cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)
            init_cfg.merge_from_list(cfg_opt)

            update_logger(init_cfg, clear_before_add=True)
            setup_seed(init_cfg.seed)
            init_cfg.federate.save_to = check_point

        # load your finetuned model (saved as xxx.ckpt)
        #    in yaml file federate.save_to
        fschatbot = FSChatBot(init_cfg)
        tokenizer = fschatbot.tokenizer
        model = fschatbot.model
        device = fschatbot.device

        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        # if not os.path.exists(
        #         os.path.join(eval_dir, "results_{}".format(
        #             init_cfg.federate.save_to))):
        #     os.makedirs(
        #         os.path.join(eval_dir,
        #                      "results_{}".format(init_cfg.federate.save_to)))

        dev_df = data['val']
        test_df = data['test']

        cors, acc, probs = eval(model, tokenizer, dev_df, test_df,
                                device, fschatbot)

        results["accuracy"].append(acc.__round__(3))
        results["data_set"].append(data_set)
        results["check_point"].append(check_point)
        results["accuracy_list"].append(cors.tolist())

    results_file = os.path.join(
        eval_dir, "accuracies_{}.json".format(
            init_cfg.federate.save_to.replace("/", "_")))
    print(results_file)
    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    data_set_list = ['result_ckpt/prefix/gpt_2_arc_c_fedprox/gpt2_arc_c_fedprox_iid_30*500_dataset_client_5.json']

    check_point_list = [
        'result_ckpt/prefix/gpt_2_arc_c_fedprox/gpt2_arc_c_fedprox_iid_30*500.ckpt',
    ]
    main(data_set_list, check_point_list)
