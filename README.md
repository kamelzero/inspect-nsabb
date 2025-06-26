# Inspect AI Agent Evaluation

## Overview

Explore the use of the Inspect AI framework and for evaluating biosecurity data.
 
* AI Security Institute - Inspect AI framework for large language model evaluations
    * https://inspect.aisi.org.uk
* Proposed Biosecurity Oversight Framework for the Future of Science
    * https://osp.od.nih.gov/wp-content/uploads/2023/03/NSABB-Final-Report-Proposed-Biosecurity-Oversight-Framework-for-the-Future-of-Science.pdf
    * Dual Use Research of Concern (DURC) categories

## Data

Construct a dataset of biosecurity papers rated by DURC.

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

A sample dataset was constructed with citations: `data/task_data.json`
Each paper corresponds to 1 or more DURC categories. 

## Experiment Design

Given the DURC data, we use a multiple-answer multiple-choice (MAMC) design.

F1 is a good metric for this, since it incorportates recall and precision.
However, `test_metrics.py` shows that the built-in `f1()` metric doesn't handle this scenario correctly, so a custom one is implemented.

## Install

```
python3.11 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

`python local_eval.py`

This outputs JSON files to `logs-local`, with F1 scores on the eval dataset for each of the models. Sample level data is also written to enable more detailed analysis.

The file currently evalutes 9 models from:
* Google
* Anthropic
* OpenAI
* Mistral

## Analysis

See: `analyze.ipynb`

* This shows the tendency of some models to answer with false positives.
* It also confirms the correctness of the our F1 metric as calculated by the evaluation framework.
* Possible extensions are discussed.