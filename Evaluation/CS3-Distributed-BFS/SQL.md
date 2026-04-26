### Sanity: checking global barriers

NVSHMEM global barriers have the same base address.
We extract it from the GPU0 track and check if we can actually see the barriers

```sql
SELECT 
  EXTRACT_ARG(arg_set_id, 'debug.gpu_id') AS gpu_id, 
  COUNT(*) AS total_barrier_signals
FROM slice 
WHERE name = 'Doorbell Control' 
  AND EXTRACT_ARG(arg_set_id, 'debug.opcode') = 20
  AND (EXTRACT_ARG(arg_set_id, 'debug.remote_addr') - (EXTRACT_ARG(arg_set_id, 'debug.gpu_id') * 8)) = 26317160960
GROUP BY 1;
```

### Barrier visualization

Run the following query on Perfetto
```sql
WITH total_gpus AS (
  SELECT COUNT(DISTINCT EXTRACT_ARG(arg_set_id, 'debug.gpu_id')) AS cnt
  FROM slice WHERE name = 'Doorbell Control'
),
sequenced_signals AS (
  SELECT 
    ts, 
    EXTRACT_ARG(arg_set_id, 'debug.gpu_id') AS gpu_id,
    (EXTRACT_ARG(arg_set_id, 'debug.remote_addr') - (EXTRACT_ARG(arg_set_id, 'debug.gpu_id') * 8)) AS base_addr,
    ROW_NUMBER() OVER (
      PARTITION BY EXTRACT_ARG(arg_set_id, 'debug.gpu_id')
      ORDER BY ts
    ) AS barrier_seq
  FROM slice
  WHERE name = 'Doorbell Control' 
    AND EXTRACT_ARG(arg_set_id, 'debug.opcode') = 20
    AND EXTRACT_ARG(arg_set_id, 'debug.remote_addr') IS NOT NULL
    AND (EXTRACT_ARG(arg_set_id, 'debug.remote_addr') - (EXTRACT_ARG(arg_set_id, 'debug.gpu_id') * 8)) = 26317160960
),
barrier_stats AS (
  SELECT
    base_addr,
    barrier_seq,
    MIN(ts) AS first_ts,
    MAX(ts) AS last_ts,
    (MAX(ts) - MIN(ts)) / 1000000.0 AS spread_ms,
    GROUP_CONCAT(gpu_id ORDER BY ts) AS arrival_order
  FROM sequenced_signals
  GROUP BY base_addr, barrier_seq
  HAVING COUNT(*) = (SELECT cnt FROM total_gpus)
)
SELECT
  bs.first_ts AS ts,
  (bs.last_ts - bs.first_ts) AS dur,
  -- Embeds the wait time and the exact order (e.g. "Order: 1,0,3,2") onto the timeline block
  'Barrier ' || bs.barrier_seq || 
  ' (Wait: ' || ROUND(bs.spread_ms, 2) || 'ms) | Order: ' || bs.arrival_order AS name
FROM barrier_stats bs
WHERE bs.spread_ms > 0.05;
```

Once the query is finished, click on `Add debug track` to visualize it.
