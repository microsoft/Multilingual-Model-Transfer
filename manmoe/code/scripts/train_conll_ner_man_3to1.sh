python train_tagging_man.py --dataset conll --langs eng deu esp --unlabeled_langs ned --dev_langs ned --model_save_file "save/conll_ner_endees2nl_$1/" --fix_emb --use_charemb "${@:2}"
python train_tagging_man.py --dataset conll --langs eng deu ned --unlabeled_langs esp --dev_langs esp --model_save_file "save/conll_ner_endenl2es_$1/" --fix_emb --use_charemb "${@:2}"
python train_tagging_man.py --dataset conll --langs eng esp ned --unlabeled_langs deu --dev_langs deu --model_save_file "save/conll_ner_enesnl2de_$1/" --fix_emb --use_charemb "${@:2}"
python train_tagging_man.py --dataset conll --langs deu esp ned --unlabeled_langs eng --dev_langs eng --model_save_file "save/conll_ner_deesnl2en_$1/" --fix_emb --use_charemb "${@:2}"
