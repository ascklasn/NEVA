import torch
from timm.models import create_model
from musk import utils, modeling
from PIL import Image
from transformers import XLMRobertaTokenizer
import pandas as pd
import os

from huggingface_hub import login
my_token = 'hf_xxxxx' # replace with your Hugging Face token
login(my_token)

def xlm_tokenizer(tokens, tokenizer, max_len=100):
    tokens = tokenizer.encode(tokens)
    
    tokens = tokens[1:-1]
    # truncate over length
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]

    tokens = [tokenizer.bos_token_id] + tokens[:] + [tokenizer.eos_token_id]
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)

    text_tokens = tokens + [tokenizer.pad_token_id] * (max_len - num_tokens)
    return text_tokens, padding_mask

device = torch.device("cuda:0")

# >>>>>>>>>>>> load model >>>>>>>>>>>> #
model_config = "musk_large_patch16_384"
model = create_model(model_config).eval()
utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
model.to(device, dtype=torch.float16)
model.eval()
# <<<<<<<<<<<< load model <<<<<<<<<<<< #


# >>>>>>>>>>> process language >>>>>>>>> #
df_report = pd.read_csv("./pathology_report_en_eval.csv")
tokenizer = XLMRobertaTokenizer("./musk/tokenizer.spm")  # todo 


for idx in range(len(df_report)):
    item = df_report.iloc[idx]

    report = item.report_en
    fname = item['case_id']
    
    # import pdb; pdb.set_trace()
    try:
        txt_ids, pad = xlm_tokenizer(report, tokenizer, max_len=100)
    except:
        txt_ids, pad = xlm_tokenizer("unknown", tokenizer, max_len=100)
        print(f"{fname}: {report}")
        os._exit(0)

    txt_ids = torch.tensor(txt_ids).unsqueeze(0)
    pad = torch.tensor(pad).unsqueeze(0)
    
    with torch.inference_mode():
        text_embeddings = model(
            text_description=txt_ids.to(device),
            padding_mask=pad.to(device),
            with_head=False, 
            out_norm=True
        )[1]  # return (vision_cls, text_cls)
    
    text_embeddings = text_embeddings.squeeze().cpu()
    torch.save(text_embeddings, f"./outputs/reports_large/{fname}.pt")

