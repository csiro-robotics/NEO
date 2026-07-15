# NeRFiller + ZoeDepth + MiDaS for Neo

This is NeRFiller implementation incorporated into Neo evaluation pipeline. All credid for the code goes to: https://github.com/ethanweber/nerfiller.
Additionally due to the bug in pytorch version 2.1.2 which is the default for nerfstudio we had to manually include ZoeDepth (https://github.com/isl-org/ZoeDepth) and MiDaS (https://github.com/isl-org/MiDaS) repos.

# Setup the environment

You need `nerfstudio` greater than version `1.0.0`. Then run,

```bash
pip install -e .
ns-install-cli
```

# References

    ```
    @inproceedings{weber2023nerfiller,
        title = {NeRFiller: Completing Scenes via Generative 3D Inpainting},
        author = {Ethan Weber and Aleksander Holynski and Varun Jampani and Saurabh Saxena and
            Noah Snavely and Abhishek Kar and Angjoo Kanazawa},
        booktitle = {CVPR},
        year = {2024},
    }
    ```

    ```
    @ARTICLE {Ranftl2022,
        author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
        title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
        journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        year    = "2022",
        volume  = "44",
        number  = "3"
    }
    ```

    ```
    @article{Ranftl2021,
        author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
        title     = {Vision Transformers for Dense Prediction},
        journal   = {ICCV},
        year      = {2021},
    }
    ```

    ```
    @article{birkl2023midas,
        title={MiDaS v3.1 -- A Model Zoo for Robust Monocular Relative Depth Estimation},
        author={Reiner Birkl and Diana Wofk and Matthias M{\"u}ller},
        journal={arXiv preprint arXiv:2307.14460},
        year={2023}
    }
    ```

    ```
    @article{bhat2023zoedepth,
    title={Zoedepth: Zero-shot transfer by combining relative and metric depth},
    author={Bhat, Shariq Farooq and Birkl, Reiner and Wofk, Diana and Wonka, Peter and M{\"u}ller, Matthias},
    journal={arXiv preprint arXiv:2302.12288},
    year={2023}
    }
    ```
