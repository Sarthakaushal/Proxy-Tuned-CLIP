# Proxy-Tuned-CLIP

Implementation of Proxy Tuning for CLIP models along with Weighted Proxy tuning with different alpha values

## Steps to start training and Proxy Training
1. Clone the repo
````bash
git clone <repo-url>
````
2. cd into the dir
````bash
cd Proxy-Tuned-Clip
````
3. Get dataset
````bash
kaggle datasets download -d apollo2506 eurosat-dataset
````
4. Run the following scripts
````bash
pip install -r requirements.txt

python3 train_expert.py
python3 train_target.py
python3 inference.py
````