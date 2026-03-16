#!/bin/bash
# Usa SOLO la GPU 0 (Massima efficienza, evita overhead multi-gpu)
export CUDA_VISIBLE_DEVICES=0

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
OUTPUT_BASE="outputs_energy_exp"
mkdir -p $OUTPUT_BASE



# --- ATTIVAZIONE AMBIENTE VIRTUAL ---
conda activate tokenskip_env
# echo " >> Ambiente virtuale tokenskip_env attivato."


# --- INIZIO CICLI ---
for SIZE in "${MODELS[@]}"; do
    echo "Configurazione modello: $SIZE"
    # LOGICA BATCH SIZE "adattiva" IN MODO CHE sia 16 per 3B e 7B e 8 per il modello 14B, così evitiamo la preemption
    if [ "$SIZE" == "14B" ]; then
        # 14B: Usiamo 8
        CURRENT_BATCH_SIZE=8
        echo " >> Setting SAFE batch size for 14B: $CURRENT_BATCH_SIZE"
    elif [ "$SIZE" == "7B" ]; then
        # 7B: Usiamo 16 (Veloce e stabile)
        CURRENT_BATCH_SIZE=16
        echo " >> Setting FAST batch size for 7B: $CURRENT_BATCH_SIZE"
    else
        # 3B: Usiamo 16 (Massima velocità)
        CURRENT_BATCH_SIZE=16
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
            sleep 5 # Attesa per il flush su disco

            # --- MODIFICA FONDAMENTALE (WARM START) ---
            # Cerchiamo il file generato da Python che contiene l'orario post-caricamento
            TIMING_FILE="TokenSkip/timing_info.json"
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

        done
    done
    

    MERGED_MODEL_DIR="${ADAPTER_PATH}/merged_static_weights"
    if [ -d "$MERGED_MODEL_DIR" ]; then
        echo " >> FINAL CLEANING: Removing merged model for ${SIZE}."
        rm -rf "$MERGED_MODEL_DIR"
    fi
done

echo " ALL EXPERIMENTS COMPLETED (hopefully) SUCCESSFULLY."