import re
import args
import json
import logging
from tqdm import tqdm
from typing import Dict
from datasets import Dataset


def extract_options(text):
    pattern = r'\b[A-Z](?:, [A-Z])*\b'
    match = re.search(pattern, text)
    if match:
        return match.group()
    else:
        logging.warning("提取选项失败！")
        return None


def verdict_accuracy(data: Dict, save: bool = False):
    score = []
    for idx, values in enumerate(zip(*data.values())):
        answer, ground_truth = None, None
        for k, v in zip(data.keys(), values):
            if k == "answer": answer = extract_options(v) 
            if k == "ground_truth": ground_truth = extract_options(v)

        if answer == ground_truth:
            score.append(1)
        else:
            score.append(0)

    print(f"Acc: {sum(score) / len(score)}")

    if save:
        data["acc"] = score
        dataset = Dataset.from_dict(data)
        dataset.to_csv(args.output_filename)


def eval(chain, retriever):
    data = {}
    contexts = []
    answer = []
    with open(args.eval_filename, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    tot_number = len(data['question'])
        
    # 实验：取k条
    exm = {}
    k = args.eval_k
    exm['question'] = data['question'][:k]
    exm['ground_truth'] = data['ground_truth'][:k]
    data = exm

    for query in tqdm(data["question"]):
        cxt = retriever.invoke(query)
        contexts.append([doc.page_content for doc in cxt])
        
        response = chain.invoke({"context": cxt, "query": query})
        answer.append(response.strip())
        
    data["answer"] = answer
    data["contexts"] = contexts
    
    logging.basicConfig(filename=args.logfile, level=logging.WARNING,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    verdict_accuracy(data, args.save_output)