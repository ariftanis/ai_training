don't delete VENV folder I don't remember what I installed with pip install before :)

use command to create gguf model from raw safetensors models

first activate venv

python -m venv venv

.\venv\Scripts\Activate

python convert_hf_to_gguf.py "input_folder" --outfile merged-model-q8_0.gguf --outtype q8_0
python convert_hf_to_gguf.py merged-model --outfile merged-model-q8_0.gguf --outtype 


python merge_model.py merged-model --outfile merged-model-q8_0.gguf --outtype q8_0