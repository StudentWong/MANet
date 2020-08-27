for weight_decay in 0.005 0.0005 0.00005
do
  for train_dataset in rgbt234
  do
    if [ $train_dataset = "rgbt234" ]; then
      val_dataset="gtot"
    elif  [ $train_dataset = "gtot" ]; then
      val_dataset="rgbt234"
    fi
    for layer in 2 3
    do
      if [ $layer = 2 ]; then
        train_batch=4
      elif  [ $layer = 3 ]; then
        train_batch=2
      fi
      python -u train_DCF.py -b $train_batch --wd $weight_decay --epochs 1 --train_data $train_dataset --val_data $val_dataset --layer $layer -s dcf_${weight_decay}_${layer}_${train_dataset}
      sleep 30
      #echo python -u train.py -s MA -I $I_dataset -wd $weight_decay --seed $((manual_seed+manual_seed))  -w logs/GA_SEED${manual_seed}_${weight_decay}_${I_dataset%.*}.pth -o logs/MA_SEED${manual_seed}_${weight_decay}_${I_dataset%.*}.pth
      #sleep 30
      python -u track_DCF.py --layer $layer --val_data $val_dataset -w dcf_${weight_decay}_${layer}_${train_dataset} -r dcf_result_${weight_decay}_${layer}_${train_dataset}
      sleep 30
    done
  done
done

