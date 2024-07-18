nohup python main.py --attacker_type SC --attacker_num 10 --defender_type ALL > output/sc.txt &
nohup python main.py --attacker_type SV --attacker_num 10 --defender_type ALL > output/sv.txt &
nohup python main.py --attacker_type MC --attacker_num 10 --defender_type ALL > output/mc.txt &
nohup python main.py --attacker_type MV --attacker_num 10 --defender_type ALL > output/mv.txt &