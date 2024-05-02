from utils import make_train_valid_dfs, build_loaders
from model import CLIPModel
from config import TrainingCFG, TextEncCFG

from transformers import DistilBertTokenizer
import torch, threading
import torch.nn.functional as F
from tqdm import tqdm
from power_usage import get_gpu_power
import pandas as pd
from sklearn.metrics import classification_report

exit_flag = threading.Event()
pow_usage = []
    # Create and start the daemon thread
daemon_thread = threading.Thread(target=get_gpu_power, args=(exit_flag, pow_usage), daemon=True)
daemon_thread.start()
_, val_df, test_df = make_train_valid_dfs()
model_dir = 'models/'
def get_text_embeddings(df, model):
    tokenizer = DistilBertTokenizer.from_pretrained(TextEncCFG.tokenizer)
    
    unique_captions = df['ClassName'].unique()
    arranged_captions = ['' for i in unique_captions]
    
    for ele in unique_captions:
        id = list(df['Label'][df.ClassName == ele])[0]
        arranged_captions[id] = ele

    encoded_captions = tokenizer(
                list(arranged_captions),
                padding=True,
                truncation=True,
                max_length=TextEncCFG.max_length
            )
    print(encoded_captions)
    final_caption_embeddings = []
    for idx in range(len(encoded_captions["input_ids"])):
        input_ids=torch.Tensor(encoded_captions["input_ids"][idx]).to(TrainingCFG.device)
        attention_mask = torch.Tensor(encoded_captions["attention_mask"][idx]).to(TrainingCFG.device)
        input_ids = input_ids.view(1, -1).to(torch.long)
        attention_mask = attention_mask.view(1, -1).to(torch.long)

        text_features = model.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
        text_embeddings = model.text_projection(text_features)
        final_caption_embeddings.append(text_embeddings)
        
    return torch.cat(final_caption_embeddings), arranged_captions

def get_image_embeddings(df, model):
    tokenizer = DistilBertTokenizer.from_pretrained(TextEncCFG.tokenizer)
    valid_loader = build_loaders(df, tokenizer, mode="valid")
    
    # model = CLIPModel().to(TrainingCFG.device)
    # model.load_state_dict(torch.load(model_path, map_location=TrainingCFG.device))
    # model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(
                                                        TrainingCFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

def get_logits(model_name:str, valid_df:pd.DataFrame):
    model = CLIPModel().to(TrainingCFG.device)
    model.load_state_dict(torch.load(model_name, map_location=TrainingCFG.device))
    model.eval()
    caption_embeddings, captions = get_text_embeddings(valid_df, model)
    image_embeddings =  get_image_embeddings(valid_df, model)
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(caption_embeddings, p=2, dim=-1)
    dot_similarity = image_embeddings_n @ text_embeddings_n.T
    return dot_similarity, captions

anti_expert_logits, captions = get_logits(model_dir+'anti-expert_resnet18.pt', test_df)
expert_model_logits, captions = get_logits(model_dir+'tuned_CLIP_resnet18_8.pt', test_df)
target_model_logits, captions = get_logits(model_dir+'base_target_model_resnet50.pt', test_df)
for alpha in [0.5, 1, 5,10]:
    target_model_logits = target_model_logits + (
        (alpha*expert_model_logits) - anti_expert_logits)

    preds = F.softmax(target_model_logits, dim =1)
    pred_classes = torch.argmax(preds, axis=1)
    y_preds = pred_classes.tolist()

    y = test_df['Label'].to_list()
    print(f'Classification Report for alpha = {alpha}')
    print(classification_report(y, y_preds, target_names=captions))
exit_flag.set()
daemon_thread.join()
print(pow_usage)

