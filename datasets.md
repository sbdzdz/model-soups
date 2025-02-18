## Instructions for downloading datasets (modified from https://github.com/mlfoundations/wise-ft)

### Step 1: Download

```bash
export DATA_LOCATION=~/data # feel free to change.
cd $DATA_LOCATION
```

#### [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)

```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar xvf imagenet-a.tar
rm imagenet-a.tar
```

#### [ImageNet-R](https://github.com/hendrycks/imagenet-r)

```bash
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar xvf imagenet-r.tar
rm imagenet-r.tar
```

#### [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

First, install and configure Kaggle:
```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials
mkdir -p ~/.kaggle

# Option 1: If you have Kaggle configured on another server:
# Run this on the server where Kaggle is working:
kaggle config view  # This will show your credentials
# Then copy the output to create kaggle.json on the new server

# Option 2: If you need new credentials:
# 1. Go to https://www.kaggle.com/settings
# 2. Scroll to "API" section
# 3. Click "Create New API Token" to download kaggle.json
# 4. Move the downloaded file:
mv /path/to/downloaded/kaggle.json ~/.kaggle/

# Set proper permissions (required)
chmod 600 ~/.kaggle/kaggle.json

# Download and extract the dataset
kaggle datasets download wanghaohan/imagenetsketch
unzip imagenetsketch.zip
rm imagenetsketch.zip
```

#### [ImageNet V2](https://github.com/modestyachts/ImageNetV2)

```bash
wget https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz
rm imagenetv2-matched-frequency.tar.gz
```

#### [ObjectNet](https://objectnet.dev/)

```bash
wget https://objectnet.dev/downloads/objectnet-1.0.zip
unzip objectnet-1.0.zip
rm objectnet-1.0.zip
```

#### ImageNet

Can be downloaded via https://www.image-net.org/download.php.
Please format for PyTorch, e.g., via https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh.

### Step 2: Check that datasts are downloaded

When running:
```bash
cd $DATA_LOCATION
ls
```
you should see (at least):
```bash
imagenet # containing train and val subfolders
imagenetv2-matched-frequency-format-val
imagenet-r
imagenet-a
sketch # imagenet-sketch
objectnet-1.0
```
