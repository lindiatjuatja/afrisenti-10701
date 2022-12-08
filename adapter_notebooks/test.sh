stdbuf -oL -eL python run_train.py > out/out.txt
stdbuf -oL -eL python run_train.py --use_en_train --translate_low_resource > out/out_en_train__translate_low_resource.txt