#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $HERE/.anaconda3/bin/activate

set -x

pip install gdown

gdown 'https://drive.google.com/uc?id=1mN-Q7zkLh6cfu5jXD72TveeP9SqIjmu7' -O face2016.zip
unzip face2016.zip
