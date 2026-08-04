# Optional HiggsTools integration

`HTs.evaluate` executes the optional HiggsTools evaluator. Jarvis-Operas does
not install `Higgs`, HBDataSet, or HSDataSet as base dependencies. Install the
Python HiggsTools package separately and provide already-pinned local dataset
directories.

The caller owns prediction construction. It supplies either a native
HiggsTools prediction dictionary or an envelope:

```python
result = registry.call(
    "HTs.evaluate",
    payload={
        "prediction": native_prediction,
        "metadata": {"model": "external-model", "prediction_schema_version": "1"},
    },
    hb_dataset_path="/datasets/HBDataSet-v1.7",
    hs_dataset_path="/datasets/HSDataSet-v1.1",
)
```

Dataset paths are selected in this order: explicit opera arguments, fields in
the payload/runtime observables, `HIGGSTOOLS_HBDATASET` or
`HIGGSTOOLS_HSDATASET`, then `dataset_defaults` (or a process-local configured
default). Paths are normalized and must already exist; this integration never
downloads or updates a dataset while evaluating a point.

The canonical result is JSON-safe and has `status` (`ok`, `invalid_input`,
`unavailable`, or `evaluation_error`), `direct_exclusion`,
`signal_measurements`, and dataset provenance. An excluded point is still a
successful evaluation: it returns `status: "ok"` with
`direct_exclusion.allowed: false`.
