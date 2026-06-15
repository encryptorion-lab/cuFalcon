# cuFalcon
This repository contains the artifact for our paper *"cuFalcon: An Adaptive Parallel GPU Implementation for High-Performance Falcon Acceleration"*.  

This project contains CUDA-accelerated implementation of the Falcon post-quantum signature scheme. It includes two parameter sets:
- `cuFalcon_512`: Falcon-512 implementation
- `cuFalcon_1024`: Falcon-1024 implementation

## Running the Code
For each parameter set, navigate into the corresponding folder and run the provided script to perform signing and verify correctness:

```bash
# For Falcon-512
cd cuFalcon_512
./run.sh

# For Falcon-1024
cd cuFalcon_1024
./run.sh
```

## License
This project (cuFalcon) is released under GPLv3 license. See [LICENSE](LICENSE) for more information.

Some files contain the modified code from [Falcon official repository](https://falcon-sign.info/impl/falcon.h.html). These codes are released under MIT License.

Some files contain the modified code from [Falcon-Mitaka](https://github.com/benlwk/Falcon-Mitaka). These codes are released under GPLv3 License.

## Citation
If you use cuFalcon in your research, please cite the following paper:
```
@article{DBLP:journals/tpds/LiWSYDZ26,
  author       = {Wenqian Li and
                  Hanyu Wei and
                  Shiyu Shen and
                  Hao Yang and
                  Wangchen Dai and
                  Yunlei Zhao},
  title        = {cuFalcon: An Adaptive Parallel {GPU} Implementation for High-Performance
                  Falcon Acceleration},
  journal      = {{IEEE} Trans. Parallel Distributed Syst.},
  volume       = {37},
  number       = {5},
  pages        = {1153--1167},
  year         = {2026},
  url          = {https://doi.org/10.1109/TPDS.2026.3675891},
  doi          = {10.1109/TPDS.2026.3675891},
  timestamp    = {Thu, 09 Apr 2026 13:01:26 +0200},
  biburl       = {https://dblp.org/rec/journals/tpds/LiWSYDZ26.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
