import torch
class CFG():
    img_data = 'EuroSAT/'
    captions_path_train = 'EuroSAT/test.csv'
    captions_path_val = 'EuroSAT/validation.csv'
    captions_path_test = 'EuroSAT/test.csv'
    size = 224
    model_path = 'models/'
    
class ExpertModelImgCFG():
    model = 'resnet18'
    pretrained = True
    trainable = True
    image_embedding = 512

class TargetModelImgCFG():
    model = 'resnet50'
    pretrained = True
    trainable = True
    image_embedding = 512
    
class TextEncCFG():
    model = 'distilbert-base-uncased'
    pretrained = True
    trainable = True
    text_embedding = 768
    tokenizer = 'distilbert-base-uncased'
    max_length = 200
    

class ProjCFG:
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

class ClipCFG():
    temperature = 1.0

class TrainingCFG():
    debug = True
    batch_size = 32
    num_workers = 2
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 8
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")