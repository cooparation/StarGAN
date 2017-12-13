#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $HERE/.anaconda3/bin/activate

set -x

pip install gdown

mkdir -p stargan_celebA/models
cd stargan_celebA/models
gdown 'https://drive.google.com/uc?id=1_38fMoR_nQ8epOOfuCfs7BJxqwwNXO3y' -O 20_4000_G.pth
