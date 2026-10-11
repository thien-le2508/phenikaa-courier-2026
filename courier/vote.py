"""Gom độ không chắc chắn: chạy 10 robot trên mọi phương án mission, mỗi robot chọn hướng có tổng xác suất cao nhất."""
from collections import defaultdict

from courier.nlp.decode import decode, hypotheses
from courier.policy import Map, predict


def vote_moves(graph, p, landmark_types=None, **kw):
    hyps = hypotheses(p, landmark_types, **kw)
    if not hyps:
        hyps = [(1.0, decode(p, landmark_types))]
    m = Map(graph)
    score = [defaultdict(float) for _ in range(10)]
    for w, mission in hyps:
        for r in range(10):
            score[r][predict(m, mission, r)] += w
    return [max(s, key=s.get) for s in score]
