#!/bin/bash
# Runs in background. Waits for training pipeline to complete, then triggers analysis.
LOG="/root/autodl-tmp/runs/plan8_pipeline.log"
MARKER="/root/autodl-tmp/runs/.plan8_complete"

echo "Post-train hook started, waiting for pipeline completion..."
echo "Log: ${LOG}"

# Wait for pipeline log to exist and complete
while [ ! -f "${LOG}" ]; do sleep 30; done

while ! grep -q "Plan8 Complete" "${LOG}" 2>/dev/null; do
    # Also check if pipeline crashed
    if ! pgrep -f "run_all.sh" > /dev/null 2>&1 && grep -q "Start:" "${LOG}"; then
        echo "WARNING: Pipeline process dead but not completed. Check log."
        break
    fi
    sleep 120
done

echo "$(date): Training pipeline detected as complete."

# Run analysis
echo "Running Step 1 analysis..."
cd /root/Mynet/RS-SAM3-p8
python3 analyze_step1.py | tee -a "${LOG}"

# Write marker
date > "${MARKER}"
echo "Analysis complete. Marker: ${MARKER}"
