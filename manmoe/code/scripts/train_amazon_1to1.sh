domains=("books" "dvd" "music")
for domain in ${domains[@]}
do
    python train_cls_man_moe.py --langs en --unlabeled_langs ja --dev_langs ja --domain "$domain" --model_save_file "save/${domain}_en2ja_$1/" --fix_emb "${@:2}"
    python train_cls_man_moe.py --langs en --unlabeled_langs fr --dev_langs fr --domain "$domain" --model_save_file "save/${domain}_en2fr_$1/" --fix_emb "${@:2}"
    python train_cls_man_moe.py --langs en --unlabeled_langs de --dev_langs de --domain "$domain" --model_save_file "save/${domain}_en2de_$1/" --fix_emb "${@:2}"
done
