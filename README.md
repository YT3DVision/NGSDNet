# When Glass Disappears at Night: A Novel NIR-RGB Multimodal Solution (TMLR2026)
The first multi-modal network, named NGSDNet, for nighttime glass surface detection.

## Abstract
Glass surface detection (GSD) has recently been attracting research interests. However,
existing GSD methods focus on modeling glass surface properties for daytime scenes only, and
can easily fail in nighttime scenes due to significant lighting discrepancies. We observe that,
due to the spectral differences between Near-Infrared (NIR) light sources and common LED
lights, NIR and RGB cameras capture complementary visual patterns (e.g., light reflections,
shadows, and edges) of glass surfaces, and cross-comparing their lighting and reflectance
properties can provide reliable cues for nighttime GSD. Inspired by this observation, we
propose a novel approach for nighttime GSD based on the multi-modal NIR and RGB image
pairs. We first construct a nighttime GSD dataset, which contains 6, 192 RGB-NIR image
pairs captured in diverse real-world nighttime scenes, with corresponding carefully-annotated
glass surface masks. We then propose a novel network for the nighttime GSD task with two
novel modules: (1) an RGB-NIR Guidance Enhancement (RNGE) module for extracting and
enriching the NIR reflectance features with the guidance of RGB reflectance features, and (2)
an RGB-NIR Fusion and Localization (RNFL) module for fusing RGB and NIR reflectance
features into glass features conditioned on the multi-modal illumination discrepancy-aware
features. Extensive experiments demonstrate that our method outperforms state-of-the-art
methods in nighttime scenes while generalizing well to daytime scenes. Our dataset and code
are available at https://github.com/YT3DVision/NGSDNet.

## Pretrain Model And Dateset

### Pretrain Model: https://mega.nz/folder/d7h2WZzR#AxxK_pvSm6mjnLqNsVdXBQ

###Dataset: https://mega.nz/folder/ZrYABIbZ#r6ezg-FP5xIDxtDkVakZQg

###Baidu Cloud link: https://pan.baidu.com/s/1F9HXIyh_q6gLXklMQ8F5OA?pwd=478c
