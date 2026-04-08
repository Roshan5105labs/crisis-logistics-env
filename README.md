# LogiFlow-RL

LogiFlow-RL is a deterministic OpenEnv benchmark for dynamic logistics routing. It simulates a real-world workload balancing problem where an agent must assign incoming freight to regional hubs while avoiding bottlenecks and preserving network stability.

The submission-ready environment lives in [crisis_logistics_env](C:\Users\rosha\crisis-logistics-env\crisis_logistics_env). The required baseline script is [inference.py](C:\Users\rosha\crisis-logistics-env\inference.py).

Quick start:

```bash
C:\Users\rosha\crisis-logistics-env\crisis_venv\Scripts\python.exe crisis_logistics_env\test_engine.py
C:\Users\rosha\crisis-logistics-env\crisis_venv\Scripts\python.exe crisis_logistics_env\train_and_evaluate.py
C:\Users\rosha\crisis-logistics-env\crisis_venv\Scripts\python.exe inference.py
```
