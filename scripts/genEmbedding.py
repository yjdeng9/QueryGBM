import os

import vertexai
from vertexai.language_models import TextEmbeddingModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

def dict_to_kv_text(d: Dict[str, Any], prefix: str) -> str:
    parts = []
    for k, v in d.items():
        if pd.isna(v):
            continue
        parts.append(f"{prefix}.{k}: {v}")
    return "\n".join(parts)

def row_to_multimodal_text(row: pd.Series,
                           image_cols: List[str],
                           genomics_cols: List[str],
                           clinical_cols: List[str]) -> str:
    """把单条记录的三类信息拼接成一个长文本（RAG/语义检索友好）"""
    img_dict = {c: row.get(c) for c in image_cols}
    gen_dict = {c: row.get(c) for c in genomics_cols}
    cli_dict = {c: row.get(c) for c in clinical_cols}

    blocks = []
    blocks.append("## Image Features")
    blocks.append(dict_to_kv_text(img_dict, "image"))

    blocks.append("\n## Genomics")
    blocks.append(dict_to_kv_text(gen_dict, "genomics"))

    blocks.append("\n## Clinical")
    blocks.append(dict_to_kv_text(cli_dict, "clinical"))

    # 你也可以在这里添加“任务标签/预后标签”等
    return "\n".join([b for b in blocks if b])


import argparse
import sys

def main():

    data_path = sys.argv[1]
    image_data_path = sys.argv[2]
    genomics_data_path = sys.argv[3]
    clinical_data_path = sys.argv[4]
    out_path = sys.argv[5]


    df = pd.read_csv(data_path)
    image_cols = pd.read_csv(image_data_path).columns
    genomics_cols = pd.read_csv(genomics_data_path).columns
    clinical_cols = pd.read_csv(clinical_data_path).columns

    texts = []
    for _, row in df.iterrows():
        t = row_to_multimodal_text(
            row=row,
            image_cols=image_cols,
            genomics_cols=genomics_cols,
            clinical_cols=clinical_cols
        )
        texts.append(t)

    vertexai.init(project="##PROJECT ID", location="us-central1")  # text-embedding-004 在 us-central1
    embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    out = []
    for t in texts:
        emb = embed_model.get_embeddings([t])[0]
        out.append(emb.values)

    out = np.array(out)
    np.save(out_path, out)



if __name__ == '__main__':
    main()