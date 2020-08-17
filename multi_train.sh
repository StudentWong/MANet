for manual_seed in 2 3 5 7 11 13
do
  for weight_decay in 0.005 0.0005 0.00005
  do
    for I_dataset in rgbt234_I.pkl rgbt234_I2.pkl
    do
      python -u train.py -s GA -I $I_dataset -wd $weight_decay --seed $manual_seed -o logs/GA_SEED${manual_seed}_${weight_decay}_${I_dataset%.*}.pth
      sleep 30
      python -u train.py -s MA -I $I_dataset -wd $weight_decay --seed $((manual_seed+manual_seed))  -w logs/GA_SEED${manual_seed}_${weight_decay}_${I_dataset%.*}.pth -o logs/MA_SEED${manual_seed}_${weight_decay}_${I_dataset%.*}.pth
      sleep 30
      python -u run_tracker.py --seed $((manual_seed+manual_seed+3))  -w logs/MA_SEED${manual_seed}_${weight_decay}_${I_dataset%.*}.pth -r MANet_RGB_T234_result_SEED${manual_seed}_${weight_decay}_${I_dataset%.*}
      sleep 30
    done
  done
done
