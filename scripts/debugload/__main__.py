import sys
from pathlib import Path
from stable_baselines3 import PPO

rootdir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(rootdir))
print(rootdir)

modelpath = Path(__file__).resolve().parent.parent / "lagfeature_return/out/fold1/model"
model = PPO.load(str(modelpath))
