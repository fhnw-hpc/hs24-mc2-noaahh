sequenceDiagram
    autonumber

    participant CPU as CPU (Main Thread)
    participant Q as Work Queue
    participant WN as Worker Thread N
    participant GPU as GPU Device
    participant K as Kernel (Reconstruct)

    %% Task addition
    CPU->>Q: Add decompositions (U, S, V) to the queue

    %% Worker threads processing tasks
    loop While queue is not empty
        WN->>Q: Pick up task (Task ID N)
        Q-->>WN: Provide task (U, S, V)

        Note over WN,GPU: Copy data (U, S, V) to GPU (async using streams)
        WN->>GPU: Transfer decompositions to device memory (stream N)

        Note over GPU,K: Launch kernel asynchronously
        GPU->>K: Start reconstruction (stream N)

        Note over K,GPU: Write reconstructed result to GPU memory
        K->>GPU: Store reconstructed matrix C (stream N)

        Note over GPU,WN: Copy result back to host memory (async using streams)
        GPU->>WN: Transfer reconstructed matrix to host memory (stream N)

        WN->>CPU: Trigger callback upon task completion
        CPU->>CPU: Save result to disk (after task completion)
    end

    %% Shutdown process
    CPU->>WN: Signal shutdown
    WN->>GPU: Release GPU resources