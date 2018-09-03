#!/bin/bash

# compare differnet regularizations
for reg in {'[eye_loss]','[lasso]','[enet]','[ridge]', '[r4rr]', '[wridge,wridge3]','[wlasso,wlasso3]'} # exclude owl from the discussion
do
    python main.py 'reg_exp' $reg
done

python main.py 'random_risk_exp'

python main.py 'expert_feature_only_exp'

