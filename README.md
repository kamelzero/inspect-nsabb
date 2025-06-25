# Inspect - AI Agent Evaluation

See: https://inspect.aisi.org.uk

See: https://osp.od.nih.gov/wp-content/uploads/2023/03/NSABB-Final-Report-Proposed-Biosecurity-Oversight-Framework-for-the-Future-of-Science.pdf

DURC Policy Scope – Categories of experiments
1. Enhance the harmful consequences of the agent or toxin;
2. Disrupt immunity or the effectiveness of an immunization against the agent or toxin
without clinical or agricultural justification;
3. Confer to the agent or toxin resistance to clinically or agriculturally useful prophylactic or
therapeutic interventions against that agent or toxin or facilitates their ability to evade
detection methodologies;
4. Increase the stability, transmissibility, or the ability to disseminate the agent or toxin;
5. Alter the host range or tropism of the agent or toxin;
6. Enhance the susceptibility of a host population to the agent or toxin; or
7. Generate or reconstitute an eradicated or extinct agent or toxin listed in the policy.

## Install

```
python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```
inspect eval nsabb_eval.py --model google/gemini-1.5-pro
```

You can also try other models (e.g., anthropic/claude-3-opus, google/gemini-1.5-pro) for comparison.