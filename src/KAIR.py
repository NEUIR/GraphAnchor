import os
import json
import argparse
import logging
import datetime
import yaml
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import backoff
from multiprocessing.pool import ThreadPool
import multiprocessing
from tqdm import tqdm
from eval import acc_score, F1_scorer, compute_exact, eval_asqa, acc_choice
from utils import seed_everything
from vllm import LLM, SamplingParams
import torch
import re
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F


def call_local(prompt_file, variable_dict):

    with open(prompt_file, "r") as fin:
        prompt = fin.read()
    if "llama" in args.model:
        model_template = "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        prompt = model_template.format(prompt=prompt.format(**variable_dict))
    if "qwen" in args.model:
        model_template = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        prompt = model_template.format(prompt=prompt.format(**variable_dict))
    
    response = llm.generate(prompt, sampling_params)[0].outputs[0].text
    return response

def passage_deduplication(passages):
    seen = set()
    deduplicated_passages = []
    for passage in passages:
        if passage not in seen:
            seen.add(passage)
            deduplicated_passages.append(passage)

    return deduplicated_passages
   
  
def get_context(data):

    text = "\n".join([f"Passage[{i}] = {doc}" for i, doc in enumerate(data)])
    return text

def extract_judgement_tag(output: str) -> str:
    """
    Extract the content of the <judgement> tag (sufficient / insufficient).
    Return a lowercase string; return None if not found.
    """
    match = re.search(r"<judgement>\s*(sufficient|insufficient)\s*</judgement>", 
                      output, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return None


def extract_index_tag(output: str) -> str | None:
    """
    Extract the content of the <index> tag.
    Supports two scenarios:
    1. <index> ... </index>
    2. <index> ...   (When the closing tag is missing, extract until the end of the text)
    """
    
    match = re.search(r"<index>\s*([\s\S]*?)\s*</index>", 
                      output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    
    match_open = re.search(r"<index>\s*([\s\S]*)", output, re.IGNORECASE)
    if match_open:
        return match_open.group(1).strip()

    return None


def extract_question_tag(output: str) -> str | None:
    """
    Extract the content of the <next_question> tag.
    Supports two scenarios:
    1. <next_question> ... </next_question>
    2. <next_question> ...   (When the closing tag is missing, retrieve up to the end of the text)
    """
    
    match = re.search(r"<next_question>\s*([\s\S]*?)\s*</next_question>", 
                      output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    
    match_open = re.search(r"<next_question>\s*([\s\S]*)", output, re.IGNORECASE)
    if match_open:
        return match_open.group(1).strip()

    return None

def save_log_to_file(logger, log_file="my_log", log_folder="logs"):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    current_date = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")
    log_file_name = f"{log_file}-{current_date}.log"
    file_handler = logging.FileHandler(os.path.join(log_folder, log_file_name))
    logger.addHandler(file_handler)


def is_directory_empty(directory_path: str) -> bool:
    try:
        return len(os.listdir(directory_path)) == 0
    except OSError:
        return False


def call_llm_template(template, variables):
    return call_llm(f"../prompts_KAIR/{LAUGUAGE}/{template}", variables)


def init_reasoning(query, refs):
    return call_llm_template("init_reasoning", {"query": query, "refs": refs})

def init_reasoning_plus(query, refs, previous_reasoning):
    return call_llm_template("init_reasoning_plus", {"query": query, "refs": refs, "previous_reasoning": previous_reasoning})

def step_reasoning(query, refs, previous_reasoning):
    return call_llm_template("step_reasoning", {"query": query, "refs": refs, "previous_reasoning": previous_reasoning})

def step_reasoning_plus(query, refs, previous_reasoning):
    return call_llm_template("step_reasoning_plus", {"query": query, "refs": refs, "previous_reasoning": previous_reasoning})


def gen_answer(query, refs):

    template = "gen_answer_index"
    return call_llm_template(template, {"refs": refs, "query": query,})

# for KAIR question answering
def gen_finalanswer(query, refs, knowledge_index):
    
    template = "gen_finalanswer"
    return call_llm_template(template, {"refs": refs, "knowledge_index": knowledge_index, "query": query,})

 
def gen_passageanswer(query, refs):
    
    template = "gen_finalanswer_passage"
    return call_llm_template(template, {"refs": refs, "query": query,})
 
def gen_indexanswer(query, knowledge_index):
    
    template = "gen_finalanswer_index"
    return call_llm_template(template, {"knowledge_index": knowledge_index, "query": query,})



def KAIR(doc_id, query, answer, top_k):
    retrieve_refs_log, llm_times, knowledgeindex_log, query_log, query_list ,top5passage_log, refine_passage, refs_plus_newretri, reasoning_log = [], 0, [], [], [], [], [], [], []
    step = 0
    reasoning_retry_need = 0
    ground_truth = answer
    retrieve_refs = retrieve(args.dataset, query=query, topk=top_k)
    refs_plus_newretri = retrieve_refs
    retrieve_refs_log.append({"retrieve_refs": retrieve_refs, "step": step, "flag": "init_retrieve_refs"})
    top5passage_log.append({"passage": retrieve_refs, "step": step, "flag": "init_passage"})

    context = get_context(refs_plus_newretri)
    initial_answer = gen_answer(query, context)
    llm_times += 1

    initial_reasoning = init_reasoning(query, context)
    judgement_status = extract_judgement_tag(initial_reasoning)
    knowledge_index = extract_index_tag(initial_reasoning)
    next_query = extract_question_tag(initial_reasoning)
    reasoning_log.append({"reasoning": initial_reasoning, "step": step, "flag": "init_reasoning"})
    knowledgeindex_log.append({"knowledge_index": knowledge_index, "step": step, "flag": "init_index"})
    
    if (next_query == None) or (judgement_status == None):
            retry_initial_reasoning = init_reasoning_plus(query, context, initial_reasoning)
            judgement_status = extract_judgement_tag(retry_initial_reasoning)
            status = judgement_status
            
            next_query = extract_question_tag(retry_initial_reasoning)
            new_query = next_query
            
            reasoning_log.append({"reasoning": retry_initial_reasoning, "step": step, "flag": "retry_initial_reasoning"})
            steps_reasoning = initial_reasoning +  retry_initial_reasoning
            reasoning_retry_need += 1
            print(f"reasoning need to be retry nums: {reasoning_retry_need}\n")
    else:
        new_query = next_query
        status = judgement_status
        steps_reasoning = initial_reasoning
    
    try:
        while ("insufficient" in status.lower()) and step < args.max_step:
            step += 1
            query_list.append(new_query)
            query_log.append({"query": new_query, "step": step, "flag": "False"})
            try:
                newretri_refs = retrieve(args.dataset, query=new_query, topk=top_k)
            except Exception as e:
                return {"id": doc_id, "question": query, "ground_truth": ground_truth, "skip": True, "skip_reason": f"retrieve_failed: {e}"}
            retrieve_refs_log.append({"retrieve_refs": newretri_refs, "step": step, "flag": "update_retrieve_refs"})

            context = get_context(newretri_refs)
            steps_reasoning = step_reasoning(query, context, steps_reasoning)
            llm_times += 1
            judgement_status = extract_judgement_tag(steps_reasoning)
            knowledge_index = extract_index_tag(steps_reasoning)
            next_query = extract_question_tag(steps_reasoning)
            reasoning_log.append({"reasoning": steps_reasoning, "step": step, "flag": "step_reasoning"})
            knowledgeindex_log.append({"knowledge_index": knowledge_index, "step": step, "flag": "updated_index"})

            if (new_query == None) or (judgement_status == None):
                retry_steps_reasoning = step_reasoning_plus(query, context, steps_reasoning)
                judgement_status = extract_judgement_tag(retry_steps_reasoning)
                next_query = extract_question_tag(retry_steps_reasoning)
                reasoning_log.append({"reasoning": retry_steps_reasoning, "step": step, "flag": "retry_step_reasoning"})
                steps_reasoning = steps_reasoning +  retry_steps_reasoning
                reasoning_retry_need += 1
                print(f"reasoning need to be retry nums: {reasoning_retry_need}\n")
                

            refs_plus_newretri = refs_plus_newretri + newretri_refs
            refs_plus_newretri = passage_deduplication(refs_plus_newretri)
            context = get_context(refs_plus_newretri)
            new_query = next_query
            status = judgement_status
    except Exception as e:
        return {"id": doc_id, "question": query, "ground_truth": ground_truth, "skip": True, "skip_reason": f"stutas_failed: {e}"}
        
        
    final_answer = gen_finalanswer(query, context, knowledge_index)
    passage_answer = gen_passageanswer(query, context)
    index_answer = gen_indexanswer(query, knowledge_index)
    llm_times += 1
        
    
    
    return {
        "id": doc_id,
        "question": query,
        "ground_truth": ground_truth,
        "initial_answer": initial_answer,
        "final_output": final_answer,
        "passage_output": passage_answer,
        "index_output": index_answer,
        "query_log": query_log,
        "knowledgeindex_log": knowledgeindex_log,
        "retrieve_ref_log": retrieve_refs_log,
        "reasoning_log": reasoning_log,
        "refine_passage_log" : top5passage_log,
    }



def process_doc_cell(idx, doc_cell, args):
    id_new, query, answer = idx, doc_cell["question"], doc_cell["answer"]

    if args.method == "base":
        prompt_path = f"../prompts_KAIR/{LAUGUAGE}"
        gen_name = args.method
        refs = retrieve(args.dataset, query=query, topk=args.retrieve_top_k)
        
        output = (
            call_llm(
                f"{prompt_path}/{gen_name}",
                {
                    "query": query,
                    "refs": get_context(refs)
                }
            )
        )

        return {"id": id_new, "query": query, "ground_truth": answer, "final_output": output, "refs":refs}
    else:
        return KAIR(id_new, query, answer, top_k=args.retrieve_top_k)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    LAUGUAGE = "en"
    seed_everything(66)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "llama3.1-8b-instruct",
            "qwen2.5-7b-instruct",
        ],
        default="qwen2.5-7b-instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--max_step", type=int, default=3, help="Maximum number of update steps"
    )
    parser.add_argument(
        "--max_fail_step", type=int, default=2, help="Maximum number of failed steps"
    )
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query",
    )
    parser.add_argument(
        "--max_top_k",
        type=int,
        default=20,
        help="Total maximum number of documents to retrieve",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "2wikimultihopqa",
            "hotpotqa",
            "musique",
            "bamboogle",
        ],
        default="musique",
        help="Dataset to use",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="KAIR",
        choices=["KAIR", "base"],
        help="Method to use",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="",
        help="Path to the file for resuming generation",
    )
    parser.add_argument(
        "--retrieve_method",
        type=str,
        default="emb",
        help="Retrieval method to use (emb: Dense Retrieval)",
    )
    args = parser.parse_args()

    with open("../config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    llm = LLM(
        model=config["model"][args.model],
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.8,
    )
    sampling_params = SamplingParams(max_tokens=768, temperature=0.1, top_p=0.9)
    call_llm = call_local

    alreadydone = 0
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    save_log_to_file(
        logger,
        log_file=f"{args.dataset}_{args.method}_{args.model}",
        log_folder="../log",
    )
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
        
    dataset_name = args.dataset
    corpus_dataset = "hotpotqa" if args.dataset == "bamboogle" else args.dataset
    vector_path = f"../data/corpus/{corpus_dataset}/{corpus_dataset}.index"

    if args.retrieve_method == "emb":
        emb_model = SentenceTransformer(
            config["model"]["bge-large-en-v1.5"], device=device
        )
        with open(f"../data/corpus/{corpus_dataset}/chunk.json", encoding="utf-8") as f:
            raw_data = json.load(f)
        vector = faiss.read_index(vector_path)

        def retrieve(_, query, topk):
            feature = emb_model.encode([query])
            _, match_id = vector.search(feature, topk)
            return [raw_data[i] for i in match_id[0]]


    formatted_time = datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")

    with open(f"../data/eval/{args.dataset}/test.json", encoding="utf-8") as f:
        qa_data = json.load(f)

    retrieve_method = args.retrieve_method

    save_path = f"../output/{args.dataset}/{retrieve_method}/{args.method}/{args.model}"
    os.makedirs(save_path, exist_ok=True)

    all_result = []
    if args.resume_path:
        with open(args.resume_path, "r", encoding="utf-8") as fin:
            resume_data = [json.loads(i) for i in fin.readlines()]
            all_result = resume_data
            filepath = args.resume_path
    else:
        resume_data = []
        filepath = (
            f"{save_path}/topk-{args.retrieve_top_k}-{formatted_time}.jsonl"
            if args.method != "KAIR"
            else f"{save_path}/topk-{args.retrieve_top_k}__max_step-{args.max_step}__max_fail_step-{args.max_fail_step}-{formatted_time}.jsonl"
        )
    logger.info(f"The predicted results will be saved in '{filepath}'.")
    last_id = len(resume_data)
    logger.info("start predicting ...")
    for i in tqdm(range(last_id, len(qa_data))):
        doc_cell = qa_data[i]
        result = process_doc_cell(i, doc_cell, args)
        if result and result.get("final_output") is not None:
            all_result.append(result)
            with open(filepath, "a", buffering=1) as fout:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            logger.warning(f"Invalid result for doc {i}: {result}")
        alreadydone += 1
        logger.info(f"Processed doc {alreadydone}")
        
    logger.info("start evaluating ...")

    predictions = [data["final_output"] for data in all_result]
    answers = [data["ground_truth"] for data in all_result]
    
    if args.method == "KAIR":
        passage_predictions = [data["passage_output"] for data in all_result if data.get("passage_output") not in [None, "", []]]
        index_predictions = [data["index_output"] for data in all_result if data.get("index_output") not in [None, "", []]]
        passage_index_answers = [data["ground_truth"] for data in all_result if data.get("passage_output") not in [None, "", []] and data.get("index_output") not in [None, "", []]]

    
    acc = acc_score(predictions, answers)
    f1 = F1_scorer(predictions, answers)
    em = compute_exact(predictions, answers)
    eval_result = {"Acc": acc, "F1": f1, "EM": em}
    
    if args.method == "KAIR":
        passage_acc = acc_score(passage_predictions, passage_index_answers)
        passage_f1 = F1_scorer(passage_predictions, passage_index_answers)
        passage_em = compute_exact(passage_predictions, passage_index_answers)
        passage_eval_result = {"passage_Acc": passage_acc, "passage_F1": passage_f1, "passage_EM": passage_em}
        
        index_acc = acc_score(index_predictions, passage_index_answers)
        index_f1 = F1_scorer(index_predictions, passage_index_answers)
        index_em = compute_exact(index_predictions, passage_index_answers)
        index_eval_result = {"index_Acc": index_acc, "index_F1": index_f1, "index_EM": index_em}
    

    if eval_result:
        with open(filepath, "a", buffering=1) as fout:
            fout.write(json.dumps(eval_result, ensure_ascii=False) + "\n")
            if args.method == "KAIR":
                fout.write(json.dumps(passage_eval_result, ensure_ascii=False) + "\n")
                fout.write(json.dumps(index_eval_result, ensure_ascii=False) + "\n")


    logger.info(f"eval result: {eval_result}")
    if args.method == "KAIR":
        logger.info(f"passage eval result: {passage_eval_result}")
        logger.info(f"index eval result: {index_eval_result}")

