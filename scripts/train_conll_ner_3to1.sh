# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

python3 train_tagging_man_moe.py --dataset conll --langs eng deu esp --unlabeled_langs ned --dev_langs ned --model_save_file "save/conll_ner_endees2nl_$1/" --fix_emb --default_emb umwe --private_hidden_size 200 --shared_hidden_size 200 --n_critic 1 "${@:2}"
python3 train_tagging_man_moe.py --dataset conll --langs eng deu ned --unlabeled_langs esp --dev_langs esp --model_save_file "save/conll_ner_endenl2es_$1/" --fix_emb --default_emb umwe --private_hidden_size 200 --shared_hidden_size 200 --n_critic 1 "${@:2}"
python3 train_tagging_man_moe.py --dataset conll --langs eng esp ned --unlabeled_langs deu --dev_langs deu --model_save_file "save/conll_ner_enesnl2de_$1/" --fix_emb --lowercase_char --default_emb umwe --private_hidden_size 200 --shared_hidden_size 200 --n_critic 1 "${@:2}"
