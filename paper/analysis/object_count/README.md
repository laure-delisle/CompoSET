# Object count

Counts detected object instances per image with OWLv2 (`google/owlv2-base-patch16-ensemble`):
score threshold 0.3, class-agnostic NMS at IoU 0.5, a fixed 116-category open-vocabulary prompt list.
The same detector settings are applied to every benchmark.

CompoSET uses the base image of the first 50 scenes (s001–s050), as reported in the paper.
Winoground, BiVLC, and SugarCrepe use 200-image samples with seed 42.

```
python paper/analysis/object_count/run_object_count.py                 # all benchmarks
python paper/analysis/object_count/run_object_count.py --only composet # one benchmark
```

Results are written to `results.json` in this folder. Re-running only processes images not already there.
