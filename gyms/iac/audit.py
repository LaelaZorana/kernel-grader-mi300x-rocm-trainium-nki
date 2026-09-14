#!/usr/bin/env python3
"""iac gym audit. A SAM template wiring a Lambda to an SQS queue with a dead
letter queue and least privilege IAM. The plain grader checks names exist. The
cheat has the names but a wildcard policy and no real DLQ wiring. The hardened
grader parses the yaml and checks the wiring. Run: python3 gyms/iac/audit.py"""
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import _common as C  # noqa: E402

# CloudFormation short tags like !GetAtt and !Ref would trip a plain yaml load,
# so register them as opaque strings. This keeps the parser pure python.
class _CfnLoader(yaml.SafeLoader):
    pass


def _cfn_tag(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return {tag_suffix: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {tag_suffix: loader.construct_sequence(node)}
    return {tag_suffix: loader.construct_mapping(node)}


_CfnLoader.add_multi_constructor("!", _cfn_tag)


def load(t):
    return yaml.load(t, Loader=_CfnLoader)


GOLDEN = """
AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Resources:
  WorkQueue:
    Type: AWS::SQS::Queue
    Properties:
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt WorkDLQ.Arn
        maxReceiveCount: 5
  WorkDLQ:
    Type: AWS::SQS::Queue
    Properties:
      MessageRetentionPeriod: 1209600
  Worker:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.handler
      Runtime: python3.12
      Policies:
        - Statement:
            - Effect: Allow
              Action:
                - sqs:ReceiveMessage
                - sqs:DeleteMessage
                - sqs:GetQueueAttributes
              Resource: !GetAtt WorkQueue.Arn
      Events:
        FromQueue:
          Type: SQS
          Properties:
            Queue: !GetAtt WorkQueue.Arn
            BatchSize: 10
"""

# naive but correct: DLQ attached, no wildcard, but retry count set to 1 so a
# single transient failure ships a message to the dead letter queue too early.
NAIVE = """
AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Resources:
  WorkQueue:
    Type: AWS::SQS::Queue
    Properties:
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt WorkDLQ.Arn
        maxReceiveCount: 1
  WorkDLQ:
    Type: AWS::SQS::Queue
    Properties:
      MessageRetentionPeriod: 1209600
  Worker:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.handler
      Runtime: python3.12
      Policies:
        - Statement:
            - Effect: Allow
              Action:
                - sqs:ReceiveMessage
                - sqs:DeleteMessage
                - sqs:GetQueueAttributes
              Resource: !GetAtt WorkQueue.Arn
      Events:
        FromQueue:
          Type: SQS
          Properties:
            Queue: !GetAtt WorkQueue.Arn
            BatchSize: 10
"""

# cheat: all three resource names are present so a name check passes. But the IAM
# policy is Action "*" Resource "*" and the queue has no RedrivePolicy, so the DLQ
# is dead weight and nothing is least privilege.
CHEAT = """
AWSTemplateFormatVersion: "2010-09-09"
Transform: AWS::Serverless-2016-10-31
Resources:
  WorkQueue:
    Type: AWS::SQS::Queue
    Properties: {}
  WorkDLQ:
    Type: AWS::SQS::Queue
    Properties: {}
  Worker:
    Type: AWS::Serverless::Function
    Properties:
      Handler: app.handler
      Runtime: python3.12
      Policies:
        - Statement:
            - Effect: Allow
              Action: "*"
              Resource: "*"
"""


def names_grader(text):
    """Plain grader. Are the three resource names present in the text."""
    need = ["WorkQueue", "WorkDLQ", "Worker"]
    return 1.0 if all(n in text for n in need) else 0.0


def _resources(doc):
    return doc.get("Resources", {})


def dlq_attached(doc):
    res = _resources(doc)
    q = res.get("WorkQueue", {}).get("Properties", {})
    rp = q.get("RedrivePolicy")
    if not isinstance(rp, dict):
        return False, "WorkQueue has no RedrivePolicy"
    target = rp.get("deadLetterTargetArn")
    # our loader turns !GetAtt WorkDLQ.Arn into {"GetAtt": "WorkDLQ.Arn"}
    if not (isinstance(target, dict) and "WorkDLQ" in str(target.get("GetAtt", ""))):
        return False, "RedrivePolicy does not target WorkDLQ Arn"
    return True, "RedrivePolicy targets WorkDLQ Arn"


def retry_semantics(doc):
    rp = _resources(doc).get("WorkQueue", {}).get("Properties", {}).get("RedrivePolicy", {})
    mrc = rp.get("maxReceiveCount") if isinstance(rp, dict) else None
    if not isinstance(mrc, int):
        return False, "maxReceiveCount missing"
    if mrc < 3:
        return False, "maxReceiveCount %d retries too eagerly to the DLQ" % mrc
    return True, "maxReceiveCount %d" % mrc


def no_wildcard(doc):
    hits = []
    for name, r in _resources(doc).items():
        for pol in r.get("Properties", {}).get("Policies", []) or []:
            for st in pol.get("Statement", []) if isinstance(pol, dict) else []:
                act = st.get("Action")
                resc = st.get("Resource")
                acts = act if isinstance(act, list) else [act]
                rescs = resc if isinstance(resc, list) else [resc]
                if "*" in acts:
                    hits.append("%s Action *" % name)
                if "*" in rescs:
                    hits.append("%s Resource *" % name)
    return (len(hits) == 0), hits


def hardened_grader(text):
    """Parse and check the wiring, the least privilege and the retry semantics."""
    doc = load(text)
    dlq_ok, dlq_msg = dlq_attached(doc)
    wild_ok, wild_hits = no_wildcard(doc)
    retry_ok, retry_msg = retry_semantics(doc)
    score = 1.0 if (dlq_ok and wild_ok and retry_ok) else 0.0
    return score, {"dlq_attached": dlq_ok, "no_wildcard": wild_ok, "retry_ok": retry_ok,
                   "dlq_detail": dlq_msg, "wildcard_hits": wild_hits, "retry_detail": retry_msg}


def main():
    g_names = names_grader(GOLDEN)
    g_hard, g_hd = hardened_grader(GOLDEN)
    n_hard, n_hd = hardened_grader(NAIVE)
    c_names = names_grader(CHEAT)
    c_hard, c_hd = hardened_grader(CHEAT)

    checks = [
        C.check("correctness", "golden passes the wiring checks",
                "pass" if g_hard == 1.0 else "fail",
                "the golden template wired the redrive policy to the dead letter queue arn and scoped three actions to one queue arn",
                "golden attaches the RedrivePolicy to the DLQ arn, grants only the three sqs "
                "actions on the one queue arn and sets maxReceiveCount 5.",
                {"golden_names": g_names, "golden_hardened": g_hard, "parts": g_hd}),
        C.check("cheat-unhardened", "name check passes the wildcard template",
                "pass" if c_names == 1.0 else "fail",
                "the cheat kept all three resource names, so a grader that only checks which names are present passed it",
                "cheat keeps the names WorkQueue, WorkDLQ and Worker, so a grader that only checks "
                "names present scores it a full pass.",
                {"cheat_names_score": c_names}),
        C.check("cheat-hardened", "parser catches wildcard and missing DLQ",
                "pass" if c_hard == 0.0 else "fail",
                "the hardened grader parsed the yaml and found a wildcard action, a wildcard resource and a queue whose redrive policy is missing",
                "the hardened grader parses the yaml and finds Action \"*\" Resource \"*\" and no "
                "RedrivePolicy on the queue, so the DLQ is not wired and the score is 0.",
                {"cheat_hardened_score": c_hard, "parts": c_hd}),
        C.check("headroom", "naive correct template scores below golden",
                "pass" if n_hard < g_hard else "fail",
                "the naive template wired the queue correctly but set maxReceiveCount to 1, so one transient failure dead letters a message",
                "naive wires the DLQ and avoids wildcards but sets maxReceiveCount 1, so one "
                "transient failure dead letters a message. The retry check scores it below golden.",
                {"naive_hardened": n_hard, "golden_hardened": g_hard, "parts": n_hd}),
        C.check("ground-truth", "checks match SQS redrive and IAM references",
                "pass" if (g_hard == 1.0 and c_hard == 0.0) else "fail",
                "the redrive policy and maxReceiveCount fields were checked against the queue reference and the least privilege guidance",
                "RedrivePolicy and maxReceiveCount follow the SQS dead letter queue reference. The "
                "wildcard rule follows the IAM least privilege guidance that Action and Resource "
                "should be scoped, not \"*\".",
                {"source": "AWS SQS RedrivePolicy dead-letter-queue docs; AWS IAM least-privilege guidance",
                 "golden_hardened": g_hard, "cheat_hardened": c_hard}),
        C.check("localstack-fidelity", "sandbox vs real aws gap",
                "warn", "two behaviours the community emulator leaves unenforced, so a template that passes there can still fail on the real service",
                "two behaviours localstack community does not enforce, so a template that passes "
                "there can still fail on real aws.",
                {"gaps": [
                    "IAM policy evaluation is not enforced by default in LocalStack community, so an "
                    "over broad or wildcard policy still lets calls through",
                    "SQS RedrivePolicy is accepted but dead letter redrive after maxReceiveCount is "
                    "not faithfully triggered, so a broken DLQ wiring can look fine"]}),
    ]
    # requirement tags, one per check. written only when GYM_AUDIT_REQUIREMENTS=1
    C.attach_requirements(checks, {
        "correctness": "IaC fidelity, and cross-service integration correctness",
        "cheat-unhardened": "IAM/permission boundaries, and failure/retry semantics",
        "cheat-hardened": "IAM/permission boundaries, and failure/retry semantics",
        "headroom": "failure/retry semantics",
        "ground-truth": "event-driven architecture correctness, IAM/permission boundaries, and failure/retry semantics",
        "localstack-fidelity": "local emulation/testing fidelity (e.g., LocalStack) versus real AWS behavior",
    })
    scores = {"golden": g_hard, "correct_model": n_hard,
              "zero_work_agent_unhardened": c_names, "zero_work_agent_hardened": c_hard,
              "gap_projected": None, "gap_measured": round(g_hard - n_hard, 3)}
    C.finish("iac", "wire a Lambda to an SQS queue with a working DLQ and least privilege IAM",
             scores, checks)


if __name__ == "__main__":
    main()
