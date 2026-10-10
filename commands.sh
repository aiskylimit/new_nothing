#1 +10
#log
#v1

# cd SpectralGuidedLearning && GPUS=5 bash project_commands_b200_gain.sh 

#2 -f-/mnt/local/aiskylimit_new_nothing/talas_vlm_embed/MMEB-evaloutputs-json-v5/ +a
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/logs/
#2 -f-/mnt/local/aiskylimit_new_nothing/OpenED/collected_logs.tar.gz +a

# nvidia-smi
# kill -9 $(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader)
# kill -9 2373844
# sleep 2
nvidia-smi

# ls OPSD/results/raw/olmo3-7b-think/grpo/

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NCCL_DEBUG=WARN

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./talas_vlm_embed/multi_gpu_v2.py

# cd ./talas_vlm_embed
# bash ./project_commands.sh

# cd ./multi-mode-distill
# bash ./project_commands.sh

# cd ./cypher-extract
# bash ./project_command.sh

cd ./OpenED
# bash gather_logs.sh all16
# tar -czf collected_logs.tar.gz collected_logs/all16/
# ls -lh collected_logs.tar.gz
targets="2540717 2531869 2508757 2520995 2503744 2524654"
declare -A seen

for pid in $targets; do
    p="$pid"

    while [[ "$p" =~ ^[0-9]+$ ]] && [ "$p" -gt 1 ]; do
        info=$(ps -o pid=,ppid=,user=,args= -p "$p")
        [ -z "$info" ] && break

        read -r cur parent user rest <<< "$info"

        # Chỉ xét process cha, không kill PID ban đầu
        if [ "$p" != "$pid" ] && [ -z "${seen[$p]+x}" ]; then
            seen[$p]=1

            case "$rest" in
                *opened*|*commands.sh*|*bash*|*sh*)
                    echo "Candidate parent: PID=$cur PPID=$parent CMD=$rest"
                    ;;
            esac
        fi

        p="$parent"
    done
done
# bash ./project_commands.sh

# cd ./opsd
# bash ./project_commands.sh

# cd ./offline_olmo7b_b200
# bash ./project_commands.sh
