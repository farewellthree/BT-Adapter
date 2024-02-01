pip install -r requirements.txt
#pip install flash-attn==1.0.7 --no-build-isolation
pip install -U openmim
mim install mmengine
mim install mmcv
cd ../../transformers
pip install -e .
cd ../../Video-ChatGPT/
