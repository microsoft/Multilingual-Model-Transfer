domains=("mystuff" "places" "calendar")
# domains=("places" "calendar")
# domains=("mystuff")
for domain in ${domains[@]}
do
    python train_tagging_man_moe.py --dataset cortana --langs en-us de-de es-es --unlabeled_langs zh-cn --dev_langs zh-cn --domain "$domain" --model_save_file "save/${domain}_endees2zh_$1/" --fix_emb "${@:2}"
    python train_tagging_man_moe.py --dataset cortana --langs en-us de-de zh-cn --unlabeled_langs es-es --dev_langs es-es --domain "$domain" --model_save_file "save/${domain}_endezh2es_$1/" --fix_emb "${@:2}"
    python train_tagging_man_moe.py --dataset cortana --langs en-us es-es zh-cn --unlabeled_langs de-de --dev_langs de-de --domain "$domain" --model_save_file "save/${domain}_eneszh2de_$1/" --fix_emb "${@:2}"
    python train_tagging_man_moe.py --dataset cortana --langs de-de es-es zh-cn --unlabeled_langs en-us --dev_langs en-us --domain "$domain" --model_save_file "save/${domain}_deeszh2en_$1/" --fix_emb "${@:2}"
done
