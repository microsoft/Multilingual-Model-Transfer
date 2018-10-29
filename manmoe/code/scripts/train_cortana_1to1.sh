domains=("mystuff" "places" "calendar")
# domains=("mystuff")
for domain in ${domains[@]}
do
    python train_tagging_man_moe.py --dataset cortana --langs en-us --unlabeled_langs zh-cn --dev_langs zh-cn --domain "$domain" --model_save_file "save/${domain}_en2zh_$1/" --fix_emb "${@:2}"
    python train_tagging_man_moe.py --dataset cortana --langs en-us --unlabeled_langs es-es --dev_langs es-es --domain "$domain" --model_save_file "save/${domain}_en2es_$1/" --fix_emb "${@:2}"
    python train_tagging_man_moe.py --dataset cortana --langs en-us --unlabeled_langs de-de --dev_langs de-de --domain "$domain" --model_save_file "save/${domain}_en2de_$1/" --fix_emb "${@:2}"
done
