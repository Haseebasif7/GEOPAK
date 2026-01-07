---
license: cc-by-sa-4.0
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - "train.csv"
    - "images/train"
  - split: test
    path:
    - "test.csv"
    - "images/test"
---
![image/png](https://cdn-uploads.huggingface.co/production/uploads/654bb2591a9e65ef2598d8c4/LbdiQQlMueyD_h5vKZKrI.png)

# OpenStreetView-5M <br><sub>The Many Roads to Global Visual Geolocation 📍🌍</sub>
**First authors:** [Guillaume Astruc](https://gastruc.github.io/), [Nicolas Dufour](https://nicolas-dufour.github.io/), [Ioannis Siglidis](https://imagine.enpc.fr/~siglidii/)  
**Second authors:** [Constantin Aronssohn](), Nacim Bouia, [Stephanie Fu](https://stephanie-fu.github.io/), [Romain Loiseau](https://romainloiseau.fr/), [Van Nguyen Nguyen](https://nv-nguyen.github.io/), [Charles Raude](https://imagine.enpc.fr/~raudec/), [Elliot Vincent](https://imagine.enpc.fr/~vincente/), Lintao XU, Hongyu Zhou  
**Last author:** [Loic Landrieu](https://loiclandrieu.com/)  
**Research Institute:** [Imagine](https://imagine.enpc.fr/), _LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-Vallée, France_  

## Introduction 🌍
[OpenStreetView-5M](https://imagine.enpc.fr/~ioannis.siglidis/osv5m/) is the first large-scale open geolocation benchmark of streetview images.  
To get a sense of the difficulty of the benchmark, you can play our [demo](https://huggingface.co/spaces/osv5m/plonk).  

### Dataset 💾
To download the datataset, run:
```python
# download the full dataset
from huggingface_hub import snapshot_download
snapshot_download(repo_id="osv5m/osv5m", local_dir="datasets/osv5m", repo_type='dataset')
```

and finally extract:
```python
import os
import zipfile
for root, dirs, files in os.walk("datasets/osv5m"):
    for file in files:
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as zip_ref:
                zip_ref.extractall(root)
                os.remove(os.path.join(root, file))
```

You can also directly load the dataset using `load_dataset`:
```python
from datasets import load_dataset
dataset = load_dataset('osv5m/osv5m', full=False)
```
where with `full` you can specify whether you want to load the complete metadata (default: `False`).

If you only want to download the test set, you can run the script below:
```python
from huggingface_hub import hf_hub_download
for i in range(5):
    hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir="datasets/OpenWorld")
    hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir="datasets/OpenWorld")
```


### Citing 💫

```bibtex
@article{osv5m,
    title = {{OpenStreetView-5M}: {T}he Many Roads to Global Visual Geolocation},
    author = {Astruc, Guillaume and Dufour, Nicolas and Siglidis, Ioannis
      and Aronssohn, Constantin and Bouia, Nacim and Fu, Stephanie and Loiseau, Romain
      and Nguyen, Van Nguyen and Raude, Charles and Vincent, Elliot and Xu, Lintao
      and Zhou, Hongyu and Landrieu, Loic},
    journal = {CVPR},
    year = {2024},
  }
```
