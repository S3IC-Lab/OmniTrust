# FACTOOL

[TOC]

> **Whether a generated output can be supported by external tools or executable evidence under the rules of its task.**

Rather than focusing on a single scenario (e.g., QA or summarization), FACTOOL targets realistic settings where model outputs are long-form, lack explicit evidence, and span heterogeneous tasks. The framework fills this gap by explicitly *using tools to collect evidence* and *reasoning over that evidence* in a unified pipeline. 

## Framework

For clarity and practical use, this project packages the four task settings studied in FACTOOL as **four independent but interface-consistent factuality checking methods**. Conceptually, you can view them as four specialized factuality detectors, each designed for a specific generation scenario.

| Method           | Target Task                  | Verification Signal     | What It Really Checks                        |
| ---------------- | ---------------------------- | ----------------------- | -------------------------------------------- |
| **KB-Factool**   | Knowledge-based QA           | Web search evidence     | Are factual statements actually true?        |
| **Code-Factool** | Code generation              | Program execution       | Does the code really work?                   |
| **Math-Factool** | Mathematical reasoning       | Executable calculations | Which step in the reasoning is wrong?        |
| **Sci-Factool**  | Scientific literature review | Scholarly metadata      | Are citations real and correctly attributed? |

The sections below describe each method from a *usage-oriented, scenario-driven* perspective.

### Method 1: KB-Factool (Knowledge-based Factuality Checking)

**Applicable scenarios** include open-domain question answering, encyclopedic content generation, and long-form factual explanations.  
**Typical risks** arise when models confidently produce incorrect or outdated facts, especially in long answers where manual verification is expensive.

KB-Factool decomposes a generated answer into *atomic factual claims*. Each claim is transformed into search queries, external evidence is retrieved via a search engine, and the model judges whether the claim is supported by the collected evidence. The final output includes both **claim-level factuality** and **response-level factuality**, allowing users to see exactly *which statements fail and why*.

### Method 2: Code-Factool (Executable Factuality for Code Generation)

**Applicable scenarios** include code assistants, automatic function generation, and programming model evaluation.  
**Typical risks** are subtle logical errors or missing edge cases: the code looks plausible but fails in execution.

Code-Factool treats the generated code as a single verifiable object and evaluates it *purely through execution*. It automatically generates test inputs and potential reference solutions, executes them to obtain pseudo-golden outputs, and compares the target code’s behavior against these outputs. If discrepancies are found, the code is marked as non-factual (incorrect). This approach mirrors real-world engineering practice: correctness is defined by behavior, not by appearance.

### Method 3: Math-Factool (Step-level Factuality for Mathematical Reasoning)

**Applicable scenarios** include math word problems, multi-step numerical reasoning, and chain-of-thought style solutions.  
**Typical risks** involve incorrect intermediate calculations that invalidate the final answer.

Math-Factool extracts each arithmetic operation from the reasoning chain and converts it into an executable check (e.g., Python expressions). These checks are run step by step. Any failed calculation immediately flags the entire response as non-factual, while also identifying the exact erroneous step. This makes Math-Factool particularly useful as an automatic verifier for mathematical reasoning pipelines.

### Method 4: Sci-Factool (Factuality Checking for Scientific Literature Reviews)

**Applicable scenarios** include literature surveys, research background writing, and citation-aware generation.  
**Typical risks** are fabricated papers, incorrect author lists, or mismatched publication years—outputs that appear academic but are factually invalid.

Sci-Factool extracts each citation as a structured tuple *(paper title, authors, publication year)* and queries scholarly databases to retrieve real metadata. Factuality is determined by matching titles and years exactly and checking whether the generated authors form a valid subset of the real author list. This makes Sci-Factool especially suitable for auditing citation reliability in research-oriented systems.


