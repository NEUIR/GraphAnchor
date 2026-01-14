import time
import os
import json
# from llama_index import SimpleDirectoryReader
import csv
from sentence_transformers import SentenceTransformer
import faiss
import argparse
import yaml
from glob import glob
from itertools import chain
from tqdm import tqdm
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from bs4 import BeautifulSoup
import random

with open("../../../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=["bge-large-en-v1.5", "gtr-t5-xxl"],
    default="bge-large-en-v1.5",
    help="Model to use",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="2wikimultihopqa",
    choices=["asqa", "strategyqa", "2wikimultihopqa", "hotpotqa", "musique", "iirc", "others"],
    help="Dataset to use",
)
parser.add_argument("--chunk_size", type=int, default=512, help="chunk size")
parser.add_argument("--chunk_overlap", type=int, default=0, help="chunk overlap")
parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
args = parser.parse_args()


def split_text(data):

    documents = []
    for record in data:
        if record["title"]:
            combined_text = record["title"] + "\n" + record["content"]
        else:
            combined_text = record["content"]
        documents.append(Document(text=combined_text))

    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)

    contents = [node.text for node in nodes]
    return contents


def build_index(embeddings, vectorstore_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, vectorstore_path)


if __name__ == "__main__":

    model = SentenceTransformer(config["model"][args.model], device=args.device)
    if args.dataset == "asqa" or args.dataset == "strategyqa":
        dataset_name = "wiki"
    else:
        dataset_name = args.dataset
    vectorstore_path = f"../../../data/corpus/{dataset_name}/{dataset_name}.index"
    contents = []
    print("loading document ...")
    start = time.time()
    if dataset_name == "wiki":
        if os.path.exists("../../../data/corpus/wiki/gtr_wikipedia_index.pkl"):
            import pickle

            with open(
                "../../../data/corpus/wiki/gtr_wikipedia_index.pkl", "rb"
            ) as file:
                embeddings = pickle.load(file)
        else:
            with open(
                "../../../data/corpus/wiki/psgs_w100.tsv", "r", encoding="utf-8"
            ) as file:
                tsv_data = csv.DictReader(file, delimiter="\t")
                raw_data = [row["title"] + "\n" + row["text"] for row in tsv_data]
            print("dataset length", len(raw_data))
            embeddings = model.encode(raw_data, batch_size=100)
    elif dataset_name == "2wikimultihopqa":
        train = json.load(open("../../../data/corpus/2wikimultihopqa/train.json", "r"))
        dev = json.load(open("../../../data/corpus/2wikimultihopqa/dev.json", "r"))
        test = json.load(open("../../../data/corpus/2wikimultihopqa/test.json", "r"))

        data = {}
        for item in tqdm(chain(train, dev, test)):
            for title, sentences in item["context"]:
                para = " ".join(sentences)
                data[para] = title
        contents = [
            {"id": i, "content": text, "title": title}
            for i, (text, title) in enumerate(data.items())
        ]
    elif dataset_name == "hotpotqa":
        import bz2
        from multiprocessing import Pool

        def process_line(line):
            data = json.loads(line)
            item = {
                "id": data["id"],
                "title": data["title"],
                "content": "".join(data["text"]),
            }
            return item

        def generate_indexing_queries_from_bz2(bz2file, dry=False):
            if dry:
                return

            with bz2.open(bz2file, "rt") as f:
                body = [process_line(line) for line in f]

            return body

        filelist = glob("../../../data/corpus/hotpotqa/*/wiki_*.bz2")

        print("Making indexing queries...")
        pool = Pool()

        for result in tqdm(
            pool.imap(generate_indexing_queries_from_bz2, filelist), total=len(filelist)
        ):
            contents.extend(result)
    elif dataset_name == "musique":
        train = [
            json.loads(line.strip())
            for line in open(
                "../../../data/corpus/musique/musique_ans_v1.0_train.jsonl"
            )
        ] + [
            json.loads(line.strip())
            for line in open(
                "../../../data/corpus/musique/musique_full_v1.0_train.jsonl"
            )
        ]
        dev = [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_ans_v1.0_dev.jsonl")
        ] + [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_full_v1.0_dev.jsonl")
        ]
        test = [
            json.loads(line.strip())
            for line in open("../../../data/corpus/musique/musique_ans_v1.0_test.jsonl")
        ] + [
            json.loads(line.strip())
            for line in open(
                "../../../data/corpus/musique/musique_full_v1.0_test.jsonl"
            )
        ]

        tot = 0
        hist = set()
        for item in tqdm(chain(train, dev, test)):
            for p in item["paragraphs"]:
                stamp = p["title"] + " " + p["paragraph_text"]
                if not stamp in hist:
                    contents.append(
                        {"id": tot, "content": p["paragraph_text"], "title": p["title"]}
                    )
                    hist.add(stamp)
                    tot += 1
    elif dataset_name == "iirc":

        raw_filepath = "../../../data/corpus/iirc/context_articles.json"

        random.seed(13370)
        with open(raw_filepath, "r") as file:
            full_data = json.load(file)

        contents = []
        tot = 0
        for title, page_html in tqdm(full_data.items()):
            page_soup = BeautifulSoup(page_html, "html.parser")
            paragraph_texts = [
                text for text in page_soup.text.split("\n")
                if text.strip() and len(text.strip().split()) > 10
            ]
            random.shuffle(paragraph_texts)

            for paragraph_index, paragraph_text in enumerate(paragraph_texts):
                contents.append({
                    "id": tot,
                    "title": title,
                    "content": paragraph_text
                })
                tot += 1
            
    elif dataset_name == "others":  # dataset 参数设置为 yourdataset
        with open("/data/groups/QY_LLM_Other/wuzhuoyang/Data/Evaluate_dataset/ultrarag_evaluation/corpus/wiki18_100w.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                contents.append({
                    "id": item["id"],
                    "content": item["contents"],
                    "title": ""  # 保持兼容性
                })

    contents = split_text(contents)
    embeddings = model.encode(contents, batch_size=600)
    with open(
        f"../../../data/corpus/{dataset_name}/chunk.json", "w", encoding="utf-8"
    ) as fout:
        json.dump(contents, fout, ensure_ascii=False)
    print("Building index ...")
    build_index(embeddings, vectorstore_path)
    end = time.time()
