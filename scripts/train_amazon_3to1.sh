domains=("books" "dvd" "music")
langs=("fr" "ja" "de")
for domain in ${domains[@]}
do
    python train_cls_man_moe.py --langs en de fr --unlabeled_langs ja --dev_langs ja --domain "$domain" --model_save_file "save/${domain}_endefr2ja_$1/" --fix_emb --model lstm --batch_size 16 --default_emb vecmap --no_charemb --n_critic 1 --lambd 0.002 "${@:2}"
    python train_cls_man_moe.py --langs en de ja --unlabeled_langs fr --dev_langs fr --domain "$domain" --model_save_file "save/${domain}_endeja2fr_$1/" --fix_emb --model lstm --batch_size 16 --default_emb vecmap  --no_charemb --n_critic 1 --lambd 0.002 "${@:2}"
    python train_cls_man_moe.py --langs en fr ja --unlabeled_langs de --dev_langs de --domain "$domain" --model_save_file "save/${domain}_enfrja2de_$1/" --fix_emb --model lstm --batch_size 16 --default_emb vecmap  --no_charemb --n_critic 1 --lambd 0.002 "${@:2}"
done
