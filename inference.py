from utils import make_train_valid_dfs, build_loaders
from model import CLIPModel
from config import TrainingCFG, TextEncCFG, CFG

from transformers import DistilBertTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report
import pandas as pd
model_dir = CFG.model_path

_, _, test_df = make_train_valid_dfs()

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
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(
                                                        TrainingCFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)

models = []#"anti-expert_resnet18.pt", "tuned_CLIP_resnet18_8.pt", 
# "base_target_model_resnet50.pt", "tuned_CLIP_resnet50_8.pt" ]
tokenizer = DistilBertTokenizer.from_pretrained(TextEncCFG.tokenizer)
valid_loader = build_loaders(test_df, tokenizer, mode="valid")

for model_name in models:
    model = CLIPModel().to(TrainingCFG.device)
    model.load_state_dict(torch.load(model_dir+model_name, map_location=TrainingCFG.device))
    model.eval()
    
    caption_embeddings, captions = get_text_embeddings(test_df, model)
    image_embeddings =  get_image_embeddings(test_df, model)
    
    # Fusing the layers to gether 
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(caption_embeddings, p=2, dim=-1)
    dot_similarity = image_embeddings_n @ text_embeddings_n.T
    dot_similarity.shape
    
    preds = F.softmax(dot_similarity, dim =1)
    pred_classes = torch.argmax(preds, axis=1)
    print(pred_classes)
    y_preds = pred_classes.tolist()
    print(f'{"#"*19}\n\nInferences for model : {model_name} \n\t Classification Report : \n')
    print(classification_report(test_df['Label'], y_preds, target_names=captions))


print( 'Implementing proxy tuning now with alpha values as 0.5, 1, 5, 10')

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

alphas = [0.5, 1, 5 ,10]
_, val_df, test_df = make_train_valid_dfs()
y = test_df['Label'].to_list()
for alpha in alphas:
    anti_expert_logits, captions = get_logits(model_dir+'anti-expert_resnet18.pt', test_df)
    expert_model_logits, _ = get_logits(model_dir+'tuned_CLIP_resnet18_8.pt', test_df)
    target_model_logits, _ = get_logits(model_dir+'base_target_model_resnet50.pt', test_df)

    target_model_logits = target_model_logits + (
        (alpha*expert_model_logits) - anti_expert_logits)
    preds = F.softmax(target_model_logits, dim =1)
    pred_classes = torch.argmax(preds, axis=1)
    y_preds = pred_classes.tolist()
    print(classification_report(y, y_preds, target_names=captions))
    