below is a rough MLOps checklist from  [Designing Machine Learning Systems - Chip Huyen](https://github.com/arjunmnath/books/blob/main/machine-learning/designing-machine-learning-systems.pdf)



## 0. Problem Framing (non-negotiable)

* [ ] Clear user persona defined
* [ ] Concrete user pain stated (not “optimize X”)
* [ ] Primary ML metric defined (e.g. AUC, RMSE, Precision@K)
* [ ] Secondary diagnostics metrics defined
* [ ] Business metric tied to model output
* [ ] Explicit latency constraint (real-time / batch / offline)
* [ ] Cost ceiling defined
* [ ] Failure mode defined (what happens when model is wrong?)

If you can’t connect model output → business decision, stop here.

---

## 1. Data Sourcing

* [ ] Data sources listed (APIs, logs, scraping, user input, etc.)
* [ ] Data ownership + legal permission checked
* [ ] Continuous collection enabled (cron / workflow / scheduler)
* [ ] Backfill strategy defined
* [ ] Data freshness requirement documented
* [ ] Raw data stored before any transformation

---

## 2. Data Validation (before touching ML)

* [ ] Schema validation (expected columns & types)
* [ ] Range checks on numeric fields
* [ ] Null / missing value thresholds defined
* [ ] Distribution checks (drift at ingestion)
* [ ] Validation automated (Great Expectations or equivalent)
* [ ] Pipeline fails hard on broken data (no silent success)

If bad data flows in quietly, your model will fail loudly later.

---

## 3. Data Storage & Versioning

* [ ] Structured data store selected (Postgres / BigQuery / etc.)
* [ ] Unstructured data store selected (S3 / GCS / etc.)
* [ ] Metadata strategy defined (IDs, timestamps, labels)
* [ ] Data versioning enabled (DVC or equivalent)
* [ ] Reproducibility guaranteed (can recreate any dataset version)
* [ ] Access controls defined

---

## 4. Feature Engineering

* [ ] Feature list documented with rationale
* [ ] Feature generation code is deterministic
* [ ] No future information leakage confirmed
* [ ] Missing value handling strategy defined
* [ ] Outlier handling strategy defined
* [ ] Scaling / normalization consistent across train & inference
* [ ] Feature selection performed (correlation / RFE / ablation)
* [ ] Feature pipeline reusable for training & inference
* [ ] Every experiment logged (what changed, why)

If train features ≠ inference features, your accuracy is fake.

---

## 5. Labeling

* [ ] Label definition is unambiguous
* [ ] Labeling guidelines written
* [ ] Label source documented (human / rules / LLM)
* [ ] Automatic labels validated manually
* [ ] Inter-annotator agreement measured (if human-labeled)
* [ ] Class imbalance measured and acknowledged
* [ ] Label noise estimated

Weak labels are fine. Undiagnosed weak labels are not.

---

## 6. Training & Evaluation

* [ ] Proper train / validation / test split (time-aware if needed)
* [ ] No data leakage across splits
* [ ] Baseline model established
* [ ] Experiment tracking enabled (MLflow / W&B)
* [ ] Hyperparameters logged
* [ ] Model artifacts versioned
* [ ] Primary metric evaluated on test set only once
* [ ] Error analysis performed
* [ ] Per-segment performance checked
* [ ] Model registry used (not random `.pkl` files)

If you keep peeking at test metrics, you’re just tuning noise.

---

## 7. Deployment

* [ ] Deployment mode chosen (real-time / batch / interactive)
* [ ] Inference code containerized (Docker)
* [ ] Same feature pipeline used as training
* [ ] CI checks for data + model + code
* [ ] Load testing done (latency & throughput)
* [ ] Rollback strategy defined
* [ ] Model version pinned in production
* [ ] Secrets handled securely

Shipping a model without rollback is gambling.

---

## 8. Monitoring & Feedback Loops

* [ ] Every prediction logged with timestamp & model version
* [ ] Input data drift monitored
* [ ] Prediction distribution monitored
* [ ] Ground truth collection strategy defined
* [ ] Production metrics computed periodically
* [ ] Alerting thresholds set
* [ ] Retraining trigger defined (time / drift / performance)
* [ ] Model performance compared across versions

No monitoring = model decay you won’t notice until users leave.

---

## 9. Orchestration

* [ ] Entire pipeline automated (Airflow / equivalent)
* [ ] Clear DAG dependencies
* [ ] Retries + failure handling defined
* [ ] Artifacts stored (data, features, models, metrics)
* [ ] Pipeline observable (logs, traces, metrics)

Manual pipelines are prototypes pretending to be systems.
