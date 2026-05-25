#!/bin/bash
LOG="/root/autodl-tmp/runs/plan8_fixup.log"
echo "Monitor started, waiting for Plan8 fixup to complete..."

while ! grep -q "Fix-up Complete" "${LOG}" 2>/dev/null; do
    sleep 300  # check every 5 min
done

echo "$(date): Training complete. Running analysis..."
cd /root/Mynet/RS-SAM3-p8
python3 analyze_step1.py | tee -a "${LOG}"
echo "$(date): Analysis done."
