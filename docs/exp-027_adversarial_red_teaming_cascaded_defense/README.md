# Exp-027: Evaluasi Ketahanan Adversarial Red-Teaming, Prompt Injection, dan Mitigasi Manipulasi Aturan Akademik

## 📌 Ringkasan Eksperimen
Eksperimen **Exp-027** menguji ketahanan arsitektur pertahanan berlapis (*Cascaded Guardrails 4-Tier*) terhadap 160 butir skenario uji terstratifikasi:
- 40 Kueri Prompt Injection & System Leakage
- 40 Kueri Manipulasi Aturan Akademik & Bypass Prasyarat
- 40 Kueri Out-of-Scope (OOD) Tugas Pemrograman Lepas
- 40 Kueri Kontrol Sah Mahasiswa (Benign Academic Formal & Slang)

## 🎯 Komparasi Varian
1. **Variant A (Baseline Raw ReAct)**: ReAct agent tanpa safety wrapper.
2. **Variant B (NeMo Alone)**: Intent rail NeMo Guardrails murni.
3. **Variant C (Lightweight Cascaded)**: Tier 1 Syntax + Tier 2 Semantic Gate + Tier 4 Parameter Rail (<35ms).
4. **Variant D (Hybrid Cascaded)**: Tier 1 + Tier 2 + NeMo + Tier 4 (Defense-in-Depth).

## 📁 Artefak
- `benchmark_summary.md`: Laporan komparatif eksekutif untuk Bab 4 Laporan TA.
- `benchmark_metrics.csv`: Data matriks evaluasi numerik tabular.
- `benchmark_results.json`: Rekap JSON metrik per varian.
- `benchmark_raw_traces.jsonl`: Jejak evaluasi lengkap 160 kueri.
