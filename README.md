# cuFalcon
This repository contains the artifact for our paper *"cuFalcon: An Adaptive Parallel GPU Implementation for High-Performance Falcon Acceleration"*.  

Authors:
- Wenqian Li, Fudan University, Shanghai, China, liwq24@m.fudan.edu.cn; 
- Hanyu Wei, Fudan University, Shanghai, China;
- Shiyu Shen, City University of Hong Kong, Hong Kong, China;
- Hao Yang, City University of Hong Kong, Hong Kong, China;
- Wangchen Dai, Sun Yat-sen University, Shenzhen, China;
- Yunlei Zhao, Fudan University, Shanghai, China.

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
