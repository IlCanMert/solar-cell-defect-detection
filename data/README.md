# ELPV Dataset Information and Licensing

## Source
- **Official Repository:** https://github.com/zae-bayern/elpv-dataset
- **Research Data Record (Jülich DATA):** https://data.fz-juelich.de/dataset.xhtml?persistentId=doi:10.26165/JUELICH-DATA/GCBNMA

## License & Usage Policy
- **Images:** Published under **CC BY-NC-SA 4.0** (Attribution-NonCommercial-ShareAlike 4.0 International). Academic attribution and license terms must be respected.
- **Loader Source Code (`elpv-dataset`):** Published under **Apache 2.0**.
- **Repository Policy:** Raw ELPV images are **not** re-uploaded to Git/GitHub in bulk. Instead, the dataset is loaded programmatically via the official `elpv-dataset` Python package (`from elpv_dataset.utils import load_dataset`), while `data/splits.csv` records reproducible train/validation/test indices.

## Included Sample Input

[`cell1857.png`](../scripts/static/cell1857.png) is an unmodified image from ELPV by Buerhop-Lutz et al. It is used as the web interface preview and the standalone CLI sample. The image corresponds to test subset entry #1576 and is shared under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Dataset Summary
- **Total images:** 2,624 grayscale electroluminescence (EL) images of single solar cells (`300 x 300` pixels, 8-bit).
- **Binary classification subset used in this project:**
  - `0.0` (`HEALTHY`): 1,508 images
  - `1.0` (`DEFECTIVE`): 715 images
  - **Total used:** 2,223 images (intermediate labels `0.33` and `0.66`/`0.67` are excluded in the initial binary formulation).

## Required Academic Citations
- Buerhop-Lutz, C., Deitsch, S., Maier, A., Gallwitz, F., Berger, S., Doll, B., Hauch, J., Camus, C., & Brabec, C. J. (2018). *A Benchmark for Visual Identification of Defective Solar Cells in Electroluminescence Imagery*. 35th European Photovoltaic Solar Energy Conference and Exhibition (EU PVSEC). https://doi.org/10.4229/35thEUPVSEC20182018-5CV.3.15
- Deitsch, S., Christlein, V., Berger, S., Buerhop-Lutz, C., Maier, A., Gallwitz, F., & Riess, C. (2019). *Automatic classification of defective photovoltaic module cells in electroluminescence images*. Solar Energy, 185, 455–468. https://doi.org/10.1016/j.solener.2019.02.067
