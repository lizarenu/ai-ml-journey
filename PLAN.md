# PLAN.md — 12-Month Learning Trajectory

This is a hypothesis, not a contract. Revisit and revise at the end of each phase.

## Overall goal

Build genuine ML competence over 12 months (June 2026 – May 2027) running parallel to the IIT-KGP Executive PG Programme in GenAI & Agentic AI. By end of year:

- Comfortable reading any ML paper and implementing core ideas
- Strong fluency in numpy, torch, and the surrounding ecosystem
- 2-3 portfolio projects that lean into my distributed systems / data platform background
- Clear sense of which sub-domain of ML I want to specialize in

The goal is NOT to "finish the course" or "learn everything." It's depth + a defensible specialization at the intersection of ML and the systems work I already do well.

## Core strategy

**Course goes broad and conceptual; I go deep and implementation-first, slightly behind.**

The IIT-KGP course covers GenAI/Agentic AI well but treats fundamentals lightly (the Foundation Bridge compresses "Deep Learning Basics" into one 3-hour session). My work fills that depth gap. When the course introduces transformers conceptually in June, I should be building micrograd and the early Karpathy videos — so by the time the course is doing RAG and fine-tuning in August/September, I've actually built a GPT from scratch and the abstractions aren't magic.

The course is the spine; my work is the muscle.

## Phases

### Phase 1: Fundamentals (June – August 2026, ~3 months)

**Goal:** Numpy/torch muscle memory + understand neural net internals deeply enough to implement them.

**Primary work:** Karpathy's "Neural Networks: Zero to Hero" series, video by video. Watch, implement, extend, log.

**Course is on:** Module 1 (Foundations of GenAI LLMs) + Module 2 (Advanced Prompting, RAG).

**By end of Phase 1, I should be able to:**
- Implement backprop from scratch and explain it cleanly
- Build a small transformer from scratch (not using `nn.Transformer`)
- Read PyTorch code without flinching at tensor manipulations
- Predict tensor shapes and gradient flow before running code

**Gap-fill weekend project:** One classical ML weekend (Kaggle-style: logistic regression, random forests, gradient boosting on a tabular dataset). Not glamorous, but necessary literacy.

### Phase 2: Domain explorations (September – November 2026, ~3 months)

**Goal:** Touch enough sub-domains to know where I want to specialize.

**Format:** 2-3 week mini-projects in different areas. Each should ship something runnable and have a writeup.

**Candidate explorations** (pick 3-4, not all):
- **Vision:** Fine-tune a vision transformer or train a small ResNet on a real dataset
- **NLP / embeddings:** Semantic search system at scale (plays to distributed systems background)
- **RL:** Implement REINFORCE or DQN on a simple environment. Especially valuable BEFORE the course covers RLHF in Module 3.
- **ML systems:** Build a small eval harness or model serving pipeline with proper observability

**Course is on:** Module 3 (Fine-Tuning Alignment) + start of Module 4 (Multimodal Agentic AI).

**By end of Phase 2, I should have:** a clear sense of which 1-2 areas I want to go deeper in.

### Phase 3: Depth + portfolio (December 2026 – May 2027, ~6 months)

**Goal:** 1-2 substantial projects that are actually impressive on a resume and concretely demonstrate ML + systems competence.

**Project criteria:**
- Plays to my distributed systems / data platform background (don't compete with new grads on ground they've already covered)
- Has a clear writeup, not just code
- Solves a real problem, even if scoped small

**Course is on:** Module 4 (Multimodal Agentic AI) + Module 5 (Deployment, Capstone).

**The IIT-KGP capstone is in Module 5 (Jan – Feb 2027).** Top 10 teams present at IIT Kharagpur. Treat this as a real target, not a checkbox. If our work in Phase 3 aligns with the capstone, even better.

**Possible Phase 3 project shapes** (decide later based on Phase 2 outcomes):
- Distributed training pipeline for a small transformer
- Production-grade RAG system with proper eval, observability, and cost tracking
- Fine-tuning infrastructure with experiment tracking and reproducibility
- Something multimodal if Module 4 sparks interest

## What this plan deliberately leaves out

- **Research / academic ML.** Not the goal. I want to be a strong ML engineer, not a researcher.
- **Massive model pretraining.** No budget, no value. Small models from scratch + fine-tuning bigger ones is the right scope.
- **Chasing every new model release.** The fundamentals don't change much; the hype cycle does. Stay focused.

## Cadence

- ~2 hours weekdays, ~3 hours weekends
- ~13 hours/week, ~675 hours over the year
- Saturday morning is the IIT-KGP live session (9am-12pm IST)
- Sunday is the bigger-task + weekly retrospective day

## Revision schedule

- **End of each phase:** review what worked, what didn't, revise next phase
- **Monthly:** check if "Current focus" in `CLAUDE.md` still reflects reality
- **Anytime motivation dips:** revisit this plan and ask whether the issue is the plan or the execution

## Course calendar reference

| Module | Dates | Topic |
|---|---|---|
| Foundation Bridge | May 17 – Jun 21, 2026 | Non-mandatory: env setup, math, Python, ML, DL, NLP basics |
| Module 1 | May 30 – Jul 18, 2026 | Foundations of GenAI LLMs (transformers, attention, prompting) |
| Module 2 | Jul 25 – Sep 5, 2026 | Advanced Prompting, RAG. **Test 1: Sep 5** |
| Module 3 | Sep 12 – Oct 31, 2026 | Fine-Tuning, PEFT, RLHF, DPO |
| Module 4 | Nov 14 – Dec 26, 2026 | Multimodal, Agents, LangChain/LangGraph. **Test 2: Dec 26** |
| Module 5 | Jan 2 – Feb 6, 2027 | Deployment, FastAPI, MCP, **Capstone** |
