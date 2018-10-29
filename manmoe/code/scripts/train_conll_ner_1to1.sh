python train_tagging_man_moe.py --dataset conll --langs eng --unlabeled_langs ned --dev_langs ned --model_save_file "save/conll_ner_en2nl_$1/" --fix_emb --use_charemb "${@:2}"
python train_tagging_man_moe.py --dataset conll --langs eng --unlabeled_langs esp --dev_langs esp --model_save_file "save/conll_ner_en2es_$1/" --fix_emb --use_charemb "${@:2}"
python train_tagging_man_moe.py --dataset conll --langs eng --unlabeled_langs deu --dev_langs deu --model_save_file "save/conll_ner_en2de_$1/" --fix_emb --use_charemb "${@:2}"
