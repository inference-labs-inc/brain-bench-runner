jobname=$1
cd notebooks/"$jobname" || exit

function run_archon_command {
  command_output=$(eval "$1")
  job_id=$(echo "$command_output" | grep -o "Recipe scheduled w/ id: [^;]*" | cut -d':' -f2 | tr -d ' ')
  archon get --poll-til-done -i "$job_id"
}

if [ -f "calibration.json" ]; then
  if ! archon create-artifact -a $jobname -i input.json -m network.onnx -c calibration.json; then
    echo "Updating artifact with calibration.json..."
    archon update-artifact -a $jobname -i input.json -m network.onnx -c calibration.json --replace
  fi
else
  if ! archon create-artifact -a $jobname -i input.json -m network.onnx; then
    echo "Updating artifact without calibration.json..."
    archon update-artifact -a $jobname -i input.json -m network.onnx --replace
  fi
fi
run_archon_command "archon job -a $jobname gen-settings"
run_archon_command "archon job -a $jobname calibrate-settings --target=resources"
run_archon_command "archon job -a $jobname get-srs"
run_archon_command "archon job -a $jobname compile-circuit"
run_archon_command "archon job -a $jobname setup"
run_archon_command "archon job -a $jobname gen-witness"

proof_times=()
for i in {1..10}; do
  proof_output=$(run_archon_command "archon job -a $jobname prove")
  proof_time=$(echo "$proof_output" | grep "proof took")
  proof_times+=("$proof_time")
done

echo "Proof times: ${proof_times[@]}"
