#!/bin/bash
# Tải trọng số từ GitHub Release v1.0 về thư mục work/.
set -e
cd "$(dirname "$0")/.."
BASE=https://github.com/thien-le2508/phenikaa-courier-2026/releases/download/v1.0
mkdir -p work/old
for f in heat_v3.pt crops_v3.pt crops_v3b.pt nlp_v3_final.pt policy_v2_final.pkl; do
  curl -L --fail -o "work/$f" "$BASE/$f"
done
curl -L --fail -o work/old/heat_strong.pt "$BASE/old_heat_strong.pt"
curl -L --fail -o work/old/crops_final.pt "$BASE/old_crops_final.pt"
echo "Đã tải xong trọng số vào work/"
