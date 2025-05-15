pip install -r requirements.txt

wget https://hazeveld.org/thesis/data.jsonl data.jsonl
wget https://hazeveld.org/thesis/flickr30k_images.tar.gz flickr30k_images.tar.gz

tar -xzf flickr30k_images.tar.gz

mkdir output

git config --global user.email "basvthazeveld@outlook.com"
git config --global user.name "Bas van 't Hazeveld"