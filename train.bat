@echo off

call conda activate unity
python main.py
REM python main.py --resume_checkpoint ppo_checkpoint_5000.pth