from servemind.evaluation.evaluator import _live_judge_one
from servemind.llm.deepseek_client import Completion


class FakeJudge:
    configured = True

    def __init__(self):
        self.user = ""

    def complete(self, *, system: str, user: str, max_tokens: int = 512):
        self.user = user
        return Completion(
            '{"helpfulness":1,"grounding":1,"boundary":1,"clarity":1,"handoff":0}',
            "deepseek-flash", 90, 20,
        )


def test_live_judge_uses_model_usage_and_redacts_ids():
    client = FakeJudge()
    judged = _live_judge_one(
        {"answer": "订单 81a6fa818d 尚未核验", "status": "clarifying",
         "grounded": False, "evidence_ids": [], "handoff": {}}, client,
    )
    assert judged["mean"] == 0.8
    assert judged["model_use"]["prompt_tokens"] == 90
    assert "81a6fa818d" not in client.user

def test_invalid_legacy_judge_returns_null_not_default_score_and_keeps_usage():
    class BadJudge(FakeJudge):
        def complete(self,**kwargs):
            return Completion('not JSON','fixture',90,20)
    judged=_live_judge_one({'answer':'尚未核验'},BadJudge())
    assert judged['mean'] is None and not judged['succeeded']
    assert judged['model_use']['prompt_tokens']==90
