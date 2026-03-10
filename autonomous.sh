#!/bin/bash

# Usa SOLO la GPU 0 (Massima efficienza, evita overhead multi-gpu)
export CUDA_VISIBLE_DEVICES=0

: '
# Controllo password Grid5000
if [ -z "$G5K_PASSWORD" ]; then
    echo "ERROR: Password Missed (G5K_PASSWORD)"
    exit 1
fi

G5K_USER="dbruno"
SITE="lyon"
NODE_NAME=$(hostname -s)
ROOT_DIR=$(pwd)

echo "========================================================"
echo " EXPERIMENT START (Optimized) ON: $NODE_NAME "
echo "========================================================"
'

# Set variabili base (poiché il blocco sopra è commentato)
NODE_NAME=$(hostname -s)
ROOT_DIR=$(pwd)

# --- CONFIGURAZIONE ---
MODELS=("3B" "7B" "14B")
BENCHMARKS=("gsm8k")
RATIOS=("1.0" "0.9" "0.8" "0.7" "0.6" "0.5") # Solo i più aggressivi per testare i limiti
BASE_MODEL_DIR="models/Qwen2.5" 
ADAPTER_BASE_DIR="models/TokenSkip-Qwen2.5"

# Cartella di Output
OUTPUT_BASE="outputs_energy_exp_final"
mkdir -p $OUTPUT_BASE

# --- ATTIVAZIONE AMBIENTE VIRTUAL ---
source /home/bd510854/first_try/tokenskip_env/bin/activate
echo " >> Ambiente virtuale tokenskip_env attivato."

# --- INIZIO CICLI ---
for SIZE in "${MODELS[@]}"; do
    echo "Configurazione modello: $SIZE"

    # ====================================================
    # LOGICA BATCH SIZE "ADATTIVA"
    # ====================================================
    if [ "$SIZE" == "14B" ]; then
        # 14B: Usiamo 8
        CURRENT_BATCH_SIZE=8
        echo " >> Setting SAFE batch size for 14B: $CURRENT_BATCH_SIZE"
    elif [ "$SIZE" == "7B" ]; then
        # 7B: Usiamo 32 (Veloce e stabile)
        CURRENT_BATCH_SIZE=32
        echo " >> Setting FAST batch size for 7B: $CURRENT_BATCH_SIZE"
    else
        # 3B: Usiamo 64 (Massima velocità)
        CURRENT_BATCH_SIZE=64
        echo " >> Setting MAX batch size for 3B: $CURRENT_BATCH_SIZE"
    fi

    # Definizione percorsi Modelli
    MODEL_REL_PATH="${BASE_MODEL_DIR}-${SIZE}-Instruct"
    MODEL_PATH="${ROOT_DIR}/${MODEL_REL_PATH}"
    
    ADAPTER_REL_PATH="${ADAPTER_BASE_DIR}-${SIZE}-Instruct-GSM8K"
    ADAPTER_PATH="${ROOT_DIR}/${ADAPTER_REL_PATH}"

    # 1. Verifica esistenza Modello Base
    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Base model directory not found at $MODEL_PATH"
        exit 1
    fi

    # 2. Verifica esistenza Adapter
    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "ERROR: Adapter directory not found at $ADAPTER_PATH"
        exit 1
    fi

    echo " >> Models found for ${SIZE}. Proceeding..."

    for BENCH in "${BENCHMARKS[@]}"; do
        # GSM8K richiede risposte lunghe
        if [ "$BENCH" == "math" ]; then MAX_TOKENS=1024; else MAX_TOKENS=512; fi

        for RATIO in "${RATIOS[@]}"; do
            echo "--------------------------------------------------------"
            echo "START: Model ${SIZE} - Ratio ${RATIO} (Batch: ${CURRENT_BATCH_SIZE})"
            echo "--------------------------------------------------------"

            # Crea cartella specifica
            EXP_DIR="${ROOT_DIR}/${OUTPUT_BASE}/${SIZE}/${BENCH}/cr_${RATIO}"
            mkdir -p "$EXP_DIR"

            # DEFINIZIONE VARIABILE MANCANTE
            RUN_NAME_ID="${SIZE}_${BENCH}_${RATIO}"

            # SKIP LOGIC CORRETTA: Se esiste già il file della GPU, salta
            if [ -s "${EXP_DIR}/${RUN_NAME_ID}_gpu.json" ]; then
                echo " >> SKIPPING: Experiment already completed (GPU File found)."
                continue
            fi

            # Configurazione Modalità (Baseline vs TokenSkip)
            if [ "$RATIO" == "1.0" ]; then
                echo " >> Mode: BASELINE (Standard Inference)"
                USE_ADAPTER_FLAG=""
                ADAPTER_PATH_ARG=""
            else
                echo " >> Mode: TOKENSKIP (LoRA Compression)"
                USE_ADAPTER_FLAG="--use_adapter"
                ADAPTER_PATH_ARG="--adapter-path ${ADAPTER_PATH}"
            fi

            # Timestamp Inizio Script (Fallback)
            # START_TIME=$(date +%s)
            
            # AVVIO MONITOR ENERGETICI PYTHON (Dalla root, prima di entrare in TokenSkip)
            echo " >> Avvio monitor PDU e GPU in background..."
            python3 monitor_pdu.py --run-name "$RUN_NAME_ID" --output-dir "$EXP_DIR" --interval 0.5 &
            PDU_PID=$!
            
            python3 monitor_gpu.py --run-name "$RUN_NAME_ID" --output-dir "$EXP_DIR" --interval 0.5 &
            GPU_PID=$!
            # ====================================================

            # Entra nella directory del codice Python DOPO aver avviato i monitor
            cd TokenSkip || exit 1

            # ESECUZIONE INFERENZA
            python evaluation.py \
            --output-dir "$EXP_DIR" \
            --model-path "${MODEL_PATH}" \
            --tokenizer-path "${MODEL_PATH}" \
            $ADAPTER_PATH_ARG \
            $USE_ADAPTER_FLAG \
            --model-size "${SIZE,,}" \
            --model-type "qwen" \
            --data-type "test" \
            --max_num_examples 100000000000000 \
            --max_new_tokens ${MAX_TOKENS} \
            --eval_batch_size $CURRENT_BATCH_SIZE \
            --temperature 0.0 \
            --seed 42 \
            --benchmark "${BENCH}" \
            --use_vllm \
            --compression_ratio ${RATIO} > "${EXP_DIR}/run_log.txt" 2>&1

            PY_EXIT_CODE=$?
            
            # Torna alla root per gestire i file JSON
            cd "$ROOT_DIR" || exit

            # 3. CHIUSURA DEI MONITOR ESTERNI
            echo " >> Inferenza terminata. Arresto monitor esterni..."
            kill -2 $PDU_PID
            kill -2 $GPU_PID
            sleep 2 # Attesa per il flush su disco
            # ====================================================

            # END_TIME=$(date +%s)

            # --- MODIFICA FONDAMENTALE (WARM START) ---
            # Cerchiamo il file generato da Python che contiene l'orario post-caricamento
            TIMING_FILE="TokenSkip/timing_info.json"
            
            : '
            if [ -f "$TIMING_FILE" ]; then
                # Usiamo Python inline per leggere il JSON in modo sicuro
                REAL_START_TIME=$(python3 -c "import json; print(json.load(open('$TIMING_FILE'))['start_inference'])")
                echo " >> ACCURATE TIMING: Using Python start time (Warm Start): $REAL_START_TIME"
            else
                # Fallback se il file non cè (es. errore o crash)
                REAL_START_TIME=$START_TIME
                echo " >> WARNING: Accurate timing file not found. Using script start time (Cold Start)."
            fi
            '
            
            # ------------------------------------------
            if [ -f "$TIMING_FILE" ]; then
                # Spostiamo il file nella cartella di output così lo hai per i grafici
                mv "$TIMING_FILE" "${EXP_DIR}/inference_timing.json"
                echo " >> Timestamp di inizio inferenza (Warm Start) salvato in ${EXP_DIR}/inference_timing.json"
            else
                echo " >> WARNING: File timing_info.json non trovato. L'analisi dovrà usare l'intero file log."
            fi

            # Gestione Errori Python
            if [ $PY_EXIT_CODE -ne 0 ]; then
                echo "ERROR: Inference crashed. Check ${EXP_DIR}/run_log.txt"
                continue
            fi
            
           echo " >> Esperimento completato."

            # echo "Inference finished. Waiting for sync..."
            # sleep 10

            # === DOWNLOAD METRICHE (WATT + BMC) ===
            # Nota: Ora usiamo REAL_START_TIME

: '
            curl -k -s -u "${G5K_USER}:${G5K_PASSWORD}" \
            "https://api.grid5000.fr/stable/sites/${SITE}/metrics?nodes=${NODE_NAME}&metrics=wattmetre_power_watt&start_time=${REAL_START_TIME}&end_time=${END_TIME}" \
            > "${EXP_DIR}/metrics_watt.json"

            curl -k -s -u "${G5K_USER}:${G5K_PASSWORD}" \
            "https://api.grid5000.fr/stable/sites/${SITE}/metrics?nodes=${NODE_NAME}&metrics=bmc_node_power_watt&start_time=${REAL_START_TIME}&end_time=${END_TIME}" \
            > "${EXP_DIR}/metrics_bmc.json"

            echo " Data saved."

            rm -f "$TIMING_FILE" # I remove the timing file to avoid confusion in the next iteration. Each run will create a new one if needed.
        done
    done
    # ---Clean: after used the static weight for each model size, I remove them) ---
    MERGED_MODEL_DIR="${ADAPTER_PATH}/merged_static_weights"
    if [ -d "$MERGED_MODEL_DIR" ]; then
        echo " >> FINAL CLEANING: Removing merged model for ${SIZE} after all ratios completed."
        rm -rf "$MERGED_MODEL_DIR"
    fi
done
'
        done
    done
    
    MERGED_MODEL_DIR="${ADAPTER_PATH}/merged_static_weights"
    if [ -d "$MERGED_MODEL_DIR" ]; then
        echo " >> FINAL CLEANING: Removing merged model for ${SIZE}."
        rm -rf "$MERGED_MODEL_DIR"
    fi
done

echo "========================================================"
echo " ALL EXPERIMENTS COMPLETED SUCCESSFULLY."
echo "========================================================"