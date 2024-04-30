from train_expert import make_train_valid_dfs, build_loaders
from model import CLIPModel
from config import TrainingCFG, CFG, TextEncCFG

from transformers import DistilBertTokenizer
import torch
from tqdm import tqdm


# Load model
def get_text_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(TextEncCFG.tokenizer)
    model = CLIPModel().to(TrainingCFG.device)
    model.load_state_dict(torch.load(model_path, map_location=TrainingCFG.device))
    model.eval()
    
    unique_captions = valid_df['ClassName'].unique()
    arranged_captions = ['' for i in unique_captions]

    for ele in unique_captions:
        id = list(valid_df['Label'][valid_df.ClassName == ele])[0]
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
        print(input_ids.shape)
        text_features = model.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
        text_embeddings = model.text_projection(text_features)
        final_caption_embeddings.append(text_embeddings)
        final_caption_embeddings = torch.cat(final_caption_embeddings)
    return model, final_caption_embeddings



def find_classes(model, text_embeddings, image, text_classes,valid_loader,n=2):
    tokenizer = DistilBertTokenizer.from_pretrained(TextEncCFG.tokenizer)
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(
                                                        TrainingCFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)
    
    
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(TrainingCFG.device) 
        for key, values in encoded_query.items()
    }
    print(batch)
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)


_, valid_df = make_train_valid_dfs()
model, captions_embeddings = get_text_embeddings(valid_df, "best.pt")