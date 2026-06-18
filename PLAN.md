# PLAN.md — 6-Month Learning Trajectory

This is a hypothesis, not a contract. Revisit and revise at the end of each phase.

## Overall goal

Build genuine AI/ML competence over 6 months (June 2026 – Dec 2026) running parallel to the IIT-KGP Executive PG Programme in GenAI & Agentic AI. By end of year:

- Strong fundamental and implementation knowledge of AI/ML concepts
- Comfortable reading any technical AI/ML paper or concept and implementing core ideas
- Strong fluency in numpy, torch, and the surrounding ecosystem
- Build AI/ML product features and deploy and maintain them 
- 2-3 portfolio projects that solve real world business issues


The goal is NOT to "finish the course" or "learn everything." It's mathematical and conceptual depth + implementation and project building proof.

## Core strategy

**the course goes broad, treat it just as the syllabus; I go deep by understanding and building**

The course is the spine; my work is the muscle.

I prioritize understanding the concept in depth, then implementing it. So I will have a strong hold on the concepts week-by-week.


**Goal:** 2-3 substantial projects that are actually impressive on a resume and concretely demonstrate AI/ML + systems + product building competence.


## What this plan deliberately leaves out

- **Research / academic ML.** Not the goal. I want to be a strong ML engineer, not a researcher.
- **Massive model pretraining.** No budget, no value. Small models from scratch + fine-tuning bigger ones is the right scope.
- **Chasing every new model release.** The fundamentals don't change much; the hype cycle does. Stay focused.

## Cadence

- ~5 hours weekdays, ~3 hours weekends
- Saturday morning is the IIT-KGP live session (9am-12pm IST)
- Sunday is weekly retrospective day

## Revision schedule

- **End of each week:** review what worked, what didn't, revise next week
- **Anytime motivation dips:** revisit this plan and ask whether the issue is the plan or the execution

## Course Syllabus 

Module 1: Foundation Models & Transformer Architecture
a. Deep learning essentials, transformer architecture (attention, tokenisation, embeddings,
positional encoding), and working with foundation models (GPT, Gemini, LLaMA,
Mistral). How to select the right model for cost, capability, and constraints.

Module 2: Advanced Prompting & RAG Systems (Weeks 7-12)
a. Systematic prompt patterns (tool-calling, retrieval-aware prompts, safety prompts, failure
handling), RAG fundamentals (chunking, retrieval architectures, vector databases), and
advanced RAG (hybrid search, re-ranking, evaluation frameworks, debugging retrieval
failures).

Module 3: LLM Fine-Tuning & Alignment (Weeks 13-18)
a. When to fine-tune versus prompt versus RAG: a decision framework.
Parameter-efficient fine-tuning (LoRA, QLoRA) on focused datasets. Three full weeks
on PEFT methods. Hands-on lab sprint covering dataset prep, training runs, evaluation,
and iteration.

Module 4: Multimodal & Agentic AI (Weeks 19-24)
a. Vision-language models and image generation pipelines. Agentic AI systems: planning,
tool use, memory, orchestration, and multi-agent workflows.

Module 5: Production Deployment & Industry Capstone (Weeks 25-30)
a. Production-grade RAG and agent orchestration at scale. Model serving (vLLM-style
concepts, FastAPI, containerisation), monitoring, and performance baselining.
Responsible AI: guardrails, privacy, hallucination handling, cost control, and governance
documentation. Industry capstone project.


| Module 1 | May 30 – Jul 18, 2026 | Foundations of GenAI LLMs (transformers, attention, prompting) |
| Module 2 | Jul 25 – Sep 5, 2026 | Advanced Prompting, RAG. **Test 1: Sep 5** |
| Module 3 | Sep 12 – Oct 31, 2026 | Fine-Tuning, PEFT, RLHF, DPO |
| Module 4 | Nov 14 – Dec 26, 2026 | Multimodal, Agents, LangChain/LangGraph. **Test 2: Dec 26** |
| Module 5 | Jan 2 – Feb 6, 2027 | Deployment, FastAPI, MCP, **Capstone** |

### Module 1 working plan (revised Jun 17, 2026)

Class is broad/fast (deep learning basics + CNN/RNN + attention already covered in 3 classes; transformer architecture is next, Sat Jun 20). We are not chasing the class pace — depth over sync. CNN gets conceptual treatment only (not core to the GenAI/agentic goal); RNN gets just enough to see the vanishing-gradient/sequential-bottleneck motivation for attention.

| Phase | Weeks | Focus | Output |
|---|---|---|---|
| 0 | Jun 18–21 | Close out deep learning essentials (backprop from scratch, gradient descent variants, regularization); light CNN/RNN treatment | NN-from-scratch implementation; RNN cell trace showing vanishing gradient |
| 1 | Jun 22–28 | Attention — Q/K/V, scaled dot-product, multi-head, from scratch in numpy then torch | Working multi-head attention implementation |
| 2 | Jun 29–Jul 5 | Full transformer block — positional encoding, layer norm, residuals, encoder-decoder vs decoder-only | Mini transformer trained on a toy char-level task |
| 3 | Jul 6–12 | Tokenization (BPE/WordPiece) + embeddings (static vs contextual, positional) | BPE tokenizer built from scratch |
| 4 | Jul 13–18 | Foundation model landscape — GPT/Gemini/LLaMA/Mistral architecture diffs (RoPE, GQA, MoE), cost/capability/constraint selection framework | Small synthesis project or case study |

This roughly follows the Karpathy zero-to-hero arc (micrograd → makemore → nanoGPT), which lines up with the existing `projects/` structure.
