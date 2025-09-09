from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='Watties as a package',
    name='Watties',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy',
        'opencv-python',
        'pyrender',
        'pytorch-lightning',
        'scikit-image',
        'smplx==0.1.28',
        'yacs',
        'detectron2 @ git+https://gitclone.com/github.com/facebookresearch/detectron2',
        'chumpy @ git+https://github.com/phangiaanh/W3Z_Packages.git@main#subdirectory=chumpy',
        'mmcv==1.3.9',
        'timm',
        'einops',
        'xtcocotools',
        'pandas',
        'open3d',
        'gradio==5.1.0',
        'pydantic==2.10.6',
    ],
    extras_require={
        'all': [
            'hydra-core',
            'hydra-submitit-launcher',
            'hydra-colorlog',
            'pyrootutils',
            'rich',
        ],
    },
)
