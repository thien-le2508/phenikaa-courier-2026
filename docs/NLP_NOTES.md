# NLP notes, round v3 (private test)

## Results (validation, 300 scenes, honest = model never saw val text or val-only phrasings)

`work/nlp_v3_holdout.pt` (train + 120k synthetic, no val3 phrasings), with map constraints from the GT landmark list:

| goal* | goal_ref | via | via_ref | urgent | fragile | exact | target |
|---|---|---|---|---|---|---|---|
| 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** |

The same model on train v3 (seen in training): every field is 1.000.

- `target` = the predicted mission, resolved on the GT map, gives exactly the same goal cell(s), via cell(s), urgent and
  fragile as the GT. This is what the policy consumes.
- goal is scored against the NLP label (None for a map-described goal unless the text names the recipient). A first version
  of the labeller had fuzzy-match false positives ("bep truong" matched "ben trong"). It was fixed with stricter Damerau
  thresholds plus a val3 role ("câu lạc bộ cầu lông"). The model had been right in all of those cases.
- The first eval after only 1500 steps already gave target 0.993.
- Final model `work/nlp_v3_final.pt`: train + val + 150k synthetic (with val3 phrasings). Train/val scores are in
  `work/nlp/run_final.stdout`. They are not a held-out number.

## Output contract (`courier/nlp/infer.py`)
```python
from courier.nlp import infer
reader = infer.load("work/nlp_v3_final.pt")          # cuda if available, else cpu
missions = infer.parse(reader, texts, landmark_type_lists)   # per text: list of types on the map (with repeats) or None
# legacy: infer.MissionReader(path).read(texts, landmark_types)  (tta= is accepted and ignored)
```
mission = `{"goal": type|None, "goal_ref": None|{"kind","anchor"}, "via": type|None, "via_ref": None|{"kind","anchor"},
"urgent": bool, "fragile": bool}`
- kinds: north/south/west/east/near/far (only when the goal type has 2 copies on the map; the near/far anchor has 1 copy),
  north_most/south_most/west_most/east_most (anchor None), anchor_near (anchor = a type on the map).
- `goal` is None only with a map kind, when the text gives no type. The policy resolves *_most as the extreme landmark
  among all landmarks, and anchor_near as the nearest landmark to the anchor by **Manhattan** distance (checked: 723/723 train,
  125/125 val. Euclidean is wrong 6+3 times).
- Rules (`rules.py`, last round's negation/polarity rules) are **OFF by default** (`rules=True` turns them on). On train they cause 8 errors.
  Example: unaccented "chu khong phai cho tra sach" reads "tra" as "trễ", negated, so urgent comes out wrong. The model alone is 100% on train and on val.
- Old checkpoints (last round's label space) are rejected with a clear error.
- `landmark_type_lists` from CV: decoding restricts goal/via/anchor to the types present. If CV drops the goal type, the decoder
  falls back to unconstrained decoding.

## What was built
- `courier/nlp/lexicon3.py`: new phrase banks. Groups: `train` (seen in train v3), `new` (written by hand, from general Vietnamese),
  `val` (seen only in val v3, so used only for the final model). The banks cover generic place nouns, *_most and anchor_near descriptions,
  extra spatial refs for typed places, correction prefixes/suffixes ("Lúc nãy nhắn nhầm là X. Đúng ra: …", "Hủy đơn giao X. Thay vào đó: …",
  "(tin trước ghi X là nhầm)" and about 45 paraphrases), contrast ("không phải X mà là G", "G chứ không phải X" and about 18 more),
  extra distractors, goal/via frames, roles, aliases, greetings and fillers.
  Last round's `lexicon.py` is used in full, including its old "val" group: that came from last round's validation, not from v3 val/test.
- `courier/nlp/generate.py`: v3 generator. Goal ref mix: none 36%, anchor_near 24%, *_most 20%, typed ref 20%. Via in 48% of samples. Map-described goals
  use a role frame in 30% of cases, which reveals the type (label = type); otherwise label = None. Corrections in 36%, contrast in 20%. Noise: lowercase,
  stripped accents, partial stripping, char typos, abbreviations, dropped punctuation.
- `courier/nlp/labels.py`: NLP labels from scenes.json. goal=None for map kinds unless a recipient role of that type is in the text
  (fuzzy Damerau match, so it tolerates typos).
- `courier/nlp/model.py`: PhoBERT-base-v2 (134.5M params incl. heads) + 8 heads. The goal head has a None class. The goal_kind head has 11 kinds + None.
- `courier/nlp/decode.py`: joint (goal, kind, anchor) decoding under map constraints.
- `courier/nlp/metrics.py`, `scripts/nlp_eval.py`: per-field, exact and target metrics, plus an error listing.
- `scripts/train_nlp.py`: training (`--max-minutes` time cap). `scripts/nlp_analyze.py`: delexicalised template counts.

## Data analysis (train v3)
- Train text = last round's train generator + new pieces. Every train alias, role and attribute phrase is in last round's lexicon "train" group.
- The new pieces in train: 3 phrasings per *_most direction, 3 per anchor_near ("nơi cạnh A nhất", "địa điểm gần A nhất", "chỗ nằm sát A nhất").
  The corrections are "Lúc nãy nhắn nhầm là X. Đúng ra:", "Hủy đơn giao X. Thay vào đó:" and "(tin trước ghi X là nhầm)". The contrasts are "không phải X mà là G" and "G chứ không phải X".
- Distractor, correction and contrast types are never the goal or via type. They can be the anchor.
- Val's novelty is mostly last round's val phrase pool ("Nhắn robot:", "Bỏ qua X, không phải ở đó.", "X không phải điểm nhận.", "Chưa đi thẳng được:",
  "kỵ va đập", "mỏng manh", "Mai giao cũng kịp", "mạn trên/trái", "sát/cách xa A"…). On top of that come "Không giao X nữa nhé, đổi lại:", "kề A nhất",
  "ở rìa trái nhất" and "sát mép dưới nhất". Val also has the via = anchor ("ghé SPO …, rồi giao tới nơi cạnh SPO nhất").
- Noise: 29% (train) / 34% (val) of texts have no diacritics, and 20-22% are all lowercase. Char-level typos are common.

## Reproduce
```bash
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 COURIER_DATA=delivery_public
python -m courier.nlp.generate --n 120000 --out work/nlp/synth_holdout.json --seed 1            # honest
python scripts/train_nlp.py --synth work/nlp/synth_holdout.json --real train --epochs 1.5 --max-minutes 45 --out work/nlp_v3_holdout.pt
python -m courier.nlp.generate --n 150000 --out work/nlp/synth_full.json --use-val --seed 3     # final
python scripts/train_nlp.py --synth work/nlp/synth_full.json --real train validation --epochs 1.5 --max-minutes 45 --no-eval --out work/nlp_v3_final.pt
python scripts/nlp_eval.py --ckpt work/nlp_v3_holdout.pt            # --split train, --no-rules, --free
```
Note: the holdout model was trained with an earlier generator. Since then the generator gained the val3 groups (used only with `--use-val`) and anchor==via (12%).

## Known weaknesses
- Val is nearly saturated, so it cannot rank further improvements. Test phrasings outside both pools (last round's val and this round's val) are the main risk.
- Typed "X gần A nhất" (kind near) and generic "chỗ gần A nhất" (anchor_near) differ only in the head noun. A new generic noun not in
  GENERIC_PLACE could be read as an alias.
- Relies on CV landmark types for the constraints. A wrong copy count can drop or keep a typed ref wrongly.
- goal for map kinds: the type is predicted only when a recipient role reveals it. It is not needed by the policy.
