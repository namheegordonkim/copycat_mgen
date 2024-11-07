# Embodied AI Seminar: Copycat for Kinematic Generative Models

## Set up the environment

Note that some requirements may have been added; I recommend installing the environment again.

```bash
conda create -y -n copycat python=3.10
```
```
conda activate copycat
```
```
pip install -r requirements.txt
```

## Put the data in the right place

Make sure that your copycat `.pkl` files are in the `data` directory.
Place your model snapshot `.pkl` files in the `models` directory (or elsewhere--you'll specify the path in the command).

## Run the code

You should be ready to go with the pretrained models. Run the following for HuMoR:

```bash
python enjoy_humor.py --humor_path [HUMOR_PATH]
```

The following is for diffusion:

```bash
python enjoy_diffusion.py --diffusion_path [DIFFUSION_PATH]
```
