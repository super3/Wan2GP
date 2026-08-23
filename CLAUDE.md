
## GPU pod launch rules (mandatory)

Money burns while pods run. Every pod launch MUST follow this or not happen:
1. REHEARSE FREE FIRST: before creating any pod, run the boot script's non-GPU
   stages locally (bash -n, then the apt/pip/clone/disk-layout steps in this
   container). A boot script that has never executed anywhere does not get a GPU.
2. KNOW THE DISK MAP: on RunPod, containerDiskInGb is the root overlay; /workspace
   is a separate small volume unless volumeInGb is set. Caches go on /.
3. WATCHDOG WITH TEETH: every pod gets a monitor that polls GraphQL latestTelemetry
   (cpu/mem/gpu) plus the live log every 60s, and AUTO-KILLS the pod itself after
   8 min of zero host telemetry with no container start, or 15 min of log stall.
   Never a watchdog that only "recommends".
4. CHECK THE BALANCE before launching and in every heartbeat: GraphQL
   myself { clientBalance currentSpendPerHr }. Stop everything below $2 remaining.
5. Results must be crash-safe: every leg/arm prints its result to the served log
   the moment it completes, never only at the end.
