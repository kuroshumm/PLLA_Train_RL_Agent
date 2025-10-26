@echo off

call conda activate unity
python main.py
REM python main.py --resume_from best_model.pth
REM python main.py --resume_from checkpoint --resume_step 20000