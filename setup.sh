pip install -r requirements.txt

wget https://hazeveld.org/thesis/data.jsonl data.jsonl
wget https://hazeveld.org/thesis/flickr30k_images.tar.gz flickr30k_images.tar.gz

tar -xzf flickr30k_images.tar.gz

mkdir output

wget https://github.com/cli/cli/releases/download/v2.72.0/gh_2.72.0_linux_amd64.deb
dpkg -i gh_2.72.0_linux_amd64.deb
gh auth login

git config --global user.email "basvthazeveld@outlook.com"
git config --global user.name "Bas van 't Hazeveld"