# Phenikaa Campus Courier 2026 · Private test: lời giải của đội

Mỗi cảnh gồm một ảnh bản đồ campus và một yêu cầu giao hàng bằng tiếng Việt. Hệ thống dự đoán **bước đi đầu tiên**
(0 = lên, 1 = xuống, 2 = trái, 3 = phải) cho 10 robot có chiến thuật ẩn khác nhau.

```
ảnh ──CV──► đồ thị (giao lộ, đoạn đường, chú giải, địa điểm, robot, thời tiết, kiểu vẽ) ─┐
                                                                                          ├─► chiến thuật 10 robot ─► 10 hướng
yêu cầu ──NLP──► (nơi giao, điểm ghé, tham chiếu bản đồ, gấp, dễ vỡ) ────────────────────┘
```

File nộp cuối: `predictions_final.json` (12.000 số, cùng thứ tự với `test_private/observations.json`).

## 1. Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `courier/cv/` | Đọc ảnh: HeatmapNet (ResNet18-FPN) tìm tâm giao lộ / chú giải / thời tiết; CropNet (ResNet18 nhiều đầu ra) phân loại giao lộ, đoạn đường, dòng chú giải; `reader.py` ghép thành đồ thị |
| `courier/nlp/` | Đọc yêu cầu: PhoBERT-base-v2 + 8 đầu ra, bộ sinh câu tổng hợp (`generate.py`, `lexicon*.py`), giải mã ràng buộc theo bản đồ (`decode.py`) |
| `courier/policy.py` | Giải tham chiếu (chính xác 100%), luật chính xác R0, đặc trưng ứng viên bước đi cho ranker |
| `courier/exact/r9.py` | Luật chính xác R9 (100% train và validation) |
| `courier/policy_features_v2.py`, `scripts/ml2_*.py`, `scripts/conditional_models.py` | Đặc trưng v2, neural inverse planner, model có điều kiện |
| `courier/style.py` | Nhận dạng kiểu vẽ ảnh (classic / night / print / sketch) |
| `scripts/predict_v2.py` | **Pipeline dự đoán cuối cùng** |
| `scripts/train_*.py`, `scripts/run_v3*.sh` | Huấn luyện |
| `scripts/policy_*.py`, `scripts/exact*.py`, `scripts/ml2_*.py` | Script phân tích và nghiên cứu chiến thuật |
| `work/` | Trọng số dùng khi dự đoán, tải từ GitHub Release (xem mục 3) |
| `docs/` | Ghi chú kỹ thuật chi tiết từng khâu |

## 2. Môi trường

- Python 3.12, GPU CUDA (đã chạy trên RTX 4050 Laptop 6 GB). Chạy CPU được nhưng chậm.
- `pip install -r requirements.txt`
- Lần đầu chạy cần tải tokenizer/cấu hình `vinai/phobert-base-v2` từ Hugging Face (công khai). Sau đó có thể đặt `HF_HUB_OFFLINE=1`.
- Dữ liệu của BTC **không** nằm trong gói. Trỏ biến `COURIER_DATA` tới thư mục `delivery_public` (chứa `train/`, `validation/`, `test_private/`).

## 3. Trọng số (`work/`)

Trọng số được đính kèm ở **GitHub Release v1.0** của repo, vì file NLP 269 MB vượt giới hạn 100 MB/file của git.
Tải toàn bộ về `work/` bằng lệnh:

```bash
bash scripts/download_weights.sh
```

| File | Kích thước | Vai trò | Dữ liệu huấn luyện |
|---|---|---|---|
| `heat_v3.pt` | 46 MB | HeatmapNet, 11.6M tham số | train v3 (fine-tune từ `old/heat_strong.pt`) |
| `crops_v3.pt`, `crops_v3b.pt` | 45 MB × 2 | CropNet, 11.2M tham số mỗi model; dự đoán lấy trung bình 2 model | train v3 (fine-tune từ `old/crops_final.pt`; v3b tăng cường ảnh mạnh hơn) |
| `nlp_v3_final.pt` | 269 MB | PhoBERT-base-v2 + 8 đầu ra, 134.5M tham số | train + validation v3 + 150k câu tổng hợp |
| `policy_v2_final.pkl` | 38 MB | Ranker HistGradientBoosting R1–R8, NIP (R4–R6), bộ phân loại kiểu vẽ | train + validation v3 |
| `old/heat_strong.pt`, `old/crops_final.pt` | 46 + 45 MB | Điểm khởi tạo (đội tự train ở vòng công khai trước) | dữ liệu công khai vòng trước |

**Tổng tham số mạng nơ-ron khi dự đoán:** 134.5M + 11.6M + 2 × 11.2M ≈ **168.5M** (< 200M). Phần còn lại là cây quyết định
(sklearn) và NIP có vài trăm trọng số. Mô hình pretrained công khai: PhoBERT-base-v2 (vinai) và ResNet18 ImageNet (torchvision).

## 4. Dự đoán (tái lập file nộp)

```bash
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 COURIER_DATA=/đường/dẫn/delivery_public
python scripts/predict_v2.py --split test_private --out predictions.json     # khoảng 10 phút trên RTX 4050
python scripts/predict_v2.py --split validation  --out val_pred.json         # kiểm tra nhanh, khoảng 3 phút
```

Đã kiểm tra: chạy gói này trên validation cho ra kết quả trùng 3000/3000 dòng với bản gốc, và chạy trên test cho ra đúng
`predictions_final.json`. Chỉ dùng ảnh, câu yêu cầu và `robot_id`; không đọc `scenes.json` khi dự đoán.
Tùy chọn `--r9 ml` dùng model học máy cho R9 thay vì luật chính xác (bản nộp trước đó).

## 5. Huấn luyện lại

```bash
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 COURIER_DATA=/đường/dẫn/delivery_public

# CV: fine-tune CropNet và HeatmapNet (xem tham số trong 2 script)
bash scripts/run_v3_train.sh      # crops_v3.pt (10 epoch) rồi heat_v3.pt (8 epoch)
bash scripts/run_v3b.sh           # crops_v3b.pt (8 epoch, tăng cường ảnh mạnh)

# NLP: sinh dữ liệu tổng hợp rồi huấn luyện (chi tiết ở mục 5.1)
python -m courier.nlp.generate --n 120000 --out work/nlp/synth_holdout.json --seed 1                 # bản trung thực
python scripts/train_nlp.py --synth work/nlp/synth_holdout.json --real train --epochs 1.5 --max-minutes 45 --out work/nlp_v3_holdout.pt
python -m courier.nlp.generate --n 150000 --out work/nlp/synth_full.json --use-val --seed 3          # bản nộp
python scripts/train_nlp.py --synth work/nlp/synth_full.json --real train validation --epochs 1.5 --max-minutes 45 --no-eval --out work/nlp_v3_final.pt

# Chiến thuật: cache đặc trưng rồi huấn luyện bundle
python scripts/policy_research.py         # work/policy_features_v1.pkl
python scripts/policy_v2_experiment.py    # work/policy_features_v2.pkl
python scripts/train_policy_best.py       # bộ phân loại kiểu vẽ (work/policy_best.pkl)
python scripts/train_policy_v2_final.py   # work/policy_v2_final.pkl
```

### 5.1 Dữ liệu tổng hợp cho NLP

Phần lớn dữ liệu huấn luyện NLP là **câu tổng hợp do đội tự sinh**, trộn với câu thật của BTC:

| Model | Câu tổng hợp | Câu thật | Mục đích |
|---|---|---|---|
| `nlp_v3_holdout.pt` | 120.000 câu, **không** chứa cách diễn đạt lấy từ validation | 2.000 câu train, lặp ×4 | đo trung thực trên validation |
| `nlp_v3_final.pt` (bản nộp) | 150.000 câu, có thêm cách diễn đạt lấy từ validation | 2.300 câu train + validation, lặp ×4 | dự đoán test |

**Cách sinh** (`courier/nlp/generate.py`, kho cụm từ `courier/nlp/lexicon.py` và `lexicon3.py`): mỗi câu được ghép từ các khung câu
có nhãn đầy đủ (nơi giao, điểm ghé, tham chiếu, gấp, dễ vỡ):

- **Khung nội dung:** nơi giao (gọi tên, tên khác, hoặc qua người nhận), điểm ghé ở 48% số câu (nhắc trước hoặc sau nơi giao),
  câu gây nhiễu ("đừng nhầm với X", "không cần ghé X", "người nhận đã rời X rồi", "hôm qua đã giao ở X rồi"),
  đính chính ở 36% ("Lúc nãy nhắn nhầm là X. Đúng ra: …", "Hủy đơn giao X. Thay vào đó: …"),
  đối chiếu ở 20% ("không phải X mà là Y", "Y chứ không phải X").
- **Tham chiếu nơi giao:** 36% không có; 24% `anchor_near` ("nơi cạnh thư viện nhất"); 20% `*_most` ("chỗ ở trên cùng bản đồ");
  20% tham chiếu theo loại (bắc / nam / tây / đông, gần / xa một mốc). Nơi giao mô tả qua bản đồ có 30% câu nêu người nhận,
  qua đó lộ ra loại địa điểm.
- **Gấp / dễ vỡ:** câu khẳng định, phủ định, mệnh lệnh phủ định, nhượng bộ ("mai giao cũng kịp"), hậu quả ("rơi là vỡ").
  Mỗi từ khóa xuất hiện ở cả hai phía nhãn để model không học tắt theo từ khóa.
- **Nhiễu:** viết thường, bỏ dấu toàn bộ hoặc một phần, lỗi gõ ở mức ký tự, viết tắt, bỏ dấu câu
  (train thật có 29% câu không dấu, 20% câu viết thường).
- **Nguồn cụm từ** chia 3 nhóm: `train` (lấy từ train), `new` (đội tự viết từ tiếng Việt thông dụng),
  `val` (lấy từ validation, chỉ bật bằng `--use-val` cho bản nộp). Nhờ đó model holdout được đo trung thực trên validation.
- **Không dùng bất kỳ câu nào của test.**

**Tái lập:**

```bash
# Bộ của bản nộp: sinh lại GIỐNG HỆT file đã dùng (đã kiểm tra 150.000/150.000 câu trùng khớp)
python -m courier.nlp.generate --n 150000 --out work/nlp/synth_full.json --use-val --seed 3
# Bộ của bản holdout: sinh bằng phiên bản generator cũ hơn, nên lệnh dưới cho bộ TƯƠNG ĐƯƠNG chứ không trùng từng câu
python -m courier.nlp.generate --n 120000 --out work/nlp/synth_holdout.json --seed 1
```

Huấn luyện trên GPU không hoàn toàn tất định. Để tái lập đúng từng dòng dự đoán, hãy dùng các trọng số trong `work/`.

## 6. Kết quả trên validation (300 cảnh, model chỉ học train)

| Khâu | Kết quả |
|---|---|
| CV (heat_v3 + 2 CropNet) | đọc đúng toàn bộ đồ thị 300/300 cảnh |
| NLP (`nlp_v3_holdout`) | đích sau khi giải đúng 300/300; mọi trường 1.000 |
| Kiểu vẽ | 300/300 |
| Chiến thuật trên đồ thị chuẩn | R0 1.000, R9 1.000; R1–R8 0.85–0.95 (xem `docs/`) |

## 7. Tuân thủ quy định

- `test_private` chỉ dùng để chạy dự đoán: không gán nhãn, không đọc câu hay xem ảnh test để viết luật, không huấn luyện trên test.
- Không gọi dịch vụ AI bên ngoài khi dự đoán.
- Tổng tham số ≈ 168.5M (< 200M).
