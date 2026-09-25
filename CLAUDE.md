
## GPU pod launch rules (mandatory)

Money burns while pods run. Every pod launch MUST follow this or not happen:
1. REHEARSE FREE FIRST: before creating any pod, run the boot script's non-GPU
   stages locally (bash -n, then the apt/pip/clone/disk-layout steps in this
   container). A boot script that has never executed anywhere does not get a GPU.
2. KNOW THE DISK MAP: on RunPod, containerDiskInGb is the root overlay; /workspace
   is a separate small volume unless volumeInGb is set. Caches go on /.
3. STAY INFORMED, DECIDE YOURSELF: every pod gets a monitor that reports full
   state (GraphQL latestTelemetry cpu/mem/gpu, account balance, live log tail)
   at least every 2 minutes, unconditionally. The agent reads every report and
   makes kill/keep decisions itself on actual data. No arbitrary auto-kill
   thresholds; no watchdog that only watches for errors and stays silent
   otherwise.
4. CHECK THE BALANCE before launching and in every heartbeat: GraphQL
   myself { clientBalance currentSpendPerHr }. Stop everything below $2 remaining.
5. Results must be crash-safe: every leg/arm prints its result to the served log
   the moment it completes, never only at the end.
