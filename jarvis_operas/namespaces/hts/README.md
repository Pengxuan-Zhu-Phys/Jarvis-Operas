# Optional HiggsTools integration

`HTs.evaluate` executes the optional HiggsTools evaluator. Jarvis-Operas does
not install `Higgs`, HBDataSet, or HSDataSet as base dependencies. The current
prototype still requires the Python HiggsTools package and already-pinned local
dataset directories.

The reviewed target design adds opt-in dataset preparation through
`HTs.prepare_datasets` and `jopera hts datasets prepare`. This is a planned API,
not yet implemented by the current prototype. The planned workflow downloads a pinned entry
from the built-in official-source catalog, verifies archive and extracted-tree
checksums, installs atomically into an immutable local cache, and then lets
`HTs.evaluate` run against that local path.

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
default). Paths are normalized and currently must already exist. In the target
design, the default remains `offline`; `if_missing` may prepare a pinned dataset
once before backend initialization, but an existing dataset is never silently
updated while evaluating points. Production scans should prepare datasets once
during job initialization and keep all workers offline.

Maintainers keep the download catalog, cache, locking, safe-extraction,
provenance, and test requirements in local internal `DESIGN_REVIEW.md` notes.
Those notes are not distributed with Git checkouts.

## Output and sample logging

HTs evaluation is file-output-free: it returns an in-memory JSON-safe mapping
and never creates result JSON, CSV, SLHA, prediction dumps, debug dumps, or its
own log file. A Jarvis host may inject its sample child logger through
`logger=`. The current prototype emits a debug completion message.

The reviewed target logging contract keeps the sample log sink and file
lifecycle with the host, permits only bounded audit events, and requires logger
failures to leave the evaluation result unchanged. These requirements and the
target modes `none`, `summary` (default), and `detailed` are not yet implemented
by the current prototype.
Summary logging records the request/schema/model identity, requested
components, particle IDs/counts, pinned dataset identities, HiggsTools version,
HiggsBounds status/allowed/selected-limit summary, HiggsSignals chi-square and
relative log-likelihood, diagnostic codes, and total elapsed time. Detailed
logging may add stage timings, cache/build information, sanitized local dataset
paths, and at most a bounded number of selected-limit records.

Under that target contract, HTs must never log the complete prediction/sample
observables/result, every BR/cross section/coupling, all applied limits or
HiggsSignals bins, environment
variables, credentials, dataset JSON, object reprs, or raw tracebacks.
Maintainers retain the event-by-event field and severity requirements in the
same local internal design notes.

The canonical result is JSON-safe and has `status` (`ok`, `invalid_input`,
`unavailable`, or `evaluation_error`), `direct_exclusion`,
`signal_measurements`, and dataset provenance. An excluded point is still a
successful evaluation: it returns `status: "ok"` with
`direct_exclusion.allowed: false`.
