# Leveraging AI in EDA: The Bottleneck Is the Data

> **The bottleneck is not LLM performance, but the data we give the LLM.**

A position paper on how EDA (Electronic Design Automation) organizations can adopt LLMs under the constraints of the domain: air-gapped networks, proprietary algorithms, contract-governed customer data, and reliability requirements that matter as much as performance.

📄 **Full paper:** [English](EDA_AI_Recommendations.md) · [한국어](EDA_AI_Recommendations_KO.md)

<p align="center">
  <img src="assets/infographic.png" alt="Infographic: discarded logs and unorganized experiment data flow into a hybrid normalization workflow (deterministic parsers for exact numbers, LLMs for context summarization and code generation, standardized input and output formats), then into a link structure connecting run lineages, recipes, and tool versions, guarded by expert verification gates, producing structured, linked, verified data assets." width="900">
</p>
<p align="center"><sub>Overview of the data-assetization pipeline proposed in Chapters 5–7. Illustration generated with AI assistance and reviewed by the author.</sub></p>

> 🇰🇷 **한국어 요약** — EDA에서 AI 활용의 병목은 LLM 성능이 아니라 LLM에 줄 수 있는 데이터의 구조화 수준이다. 수십 년간 쌓였다 지워지는 실험 리포트와 로그를 정규화·연결·검증해 자산으로 만들자는 제언이다. 전문은 [한국어 버전](EDA_AI_Recommendations_KO.md)에서 읽을 수 있다.

## TL;DR

- **The domain gap is about knowledge, not intelligence.** General-purpose LLMs know semiconductor fundamentals but are unlikely to have reliably learned tool-internal assumptions, version-specific behavior, option interactions, or customer design context.
- **The dangerous failure is not hallucination.** It is output that is syntactically valid and runs, yet is engineeringly meaningless — which shifts the burden back onto expert verification.
- **No single approach wins.** The paper compares fine-tuning, RAG, LLM Wiki, and agent/MCP tool calling against EDA constraints and recommends a hybrid: *facts via RAG and tool calls, refined know-how via Wiki, and only format and idioms via lightweight fine-tuning.*
- **The real asset is already in-house.** Decades of experiment reports and logs are routinely deleted. Normalizing them into run records and linking lineage, recipes, QoR, tool versions, and design context turns them into evidence an LLM can ground on.
- **The loop needs a verification gate.** Without expert verification, a feedback loop of AI outputs becomes a data-contamination loop. Verification should be tiered so expert time goes where only experts can judge.

**Start here:** define a run-record schema for one flow stage and one block, attach a parser that extracts records before logs are discarded (where contracts permit), and measure the pilot.

## Contents of the paper

| Chapter | Topic |
| --- | --- |
| 1–3 | Structural constraints of EDA, AI adoption so far, and the domain gap |
| 4 | Fine-tuning vs. RAG vs. LLM Wiki vs. Agent/MCP, with a comparison table |
| 5 | Data assetization: retention policy, LLM/parser role split, an example run-record schema (YAML) |
| 6 | Link structure, a worked query scenario, recipes as assets, customer boundaries |
| 7 | The positive feedback loop and tiered verification |
| 8 | Conclusion and a measurable first step |

The paper cites 32 sources, each labeled by evidence type ([Paper], [Vendor], [Commentary], etc.). All identifiers, versions, and numbers in the schema examples are invented for illustration.

## Versions

| Version | Date | Notes |
| --- | --- | --- |
| 1.0 | 2026-10-18 | First public release, English and Korean (internal draft v31) |

## Feedback

Corrections, counter-evidence, and practitioner experience are welcome via [Issues](../../issues). Please do not post confidential, customer, or employer information.

## How to cite

> Kim, Kyoungman. *Recommendations on Leveraging AI in the EDA Domain — The Bottleneck Is Not LLM Performance, but the Data We Give the LLM.* Version 1.0, 2026. GitHub repository.

A machine-readable [CITATION.cff](CITATION.cff) is included; GitHub shows it as **"Cite this repository"** in the sidebar.

## License

The text of this paper is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE). You may share and adapt it with attribution. Cited third-party works remain under their own terms.

## Disclaimer

Views expressed here are my own and do not represent my employer or any company mentioned. This work is limited to cited public sources, original analysis, and hypothetical examples created for illustration; no endorsement or affiliation is implied. Product and company names are used solely for identification and remain the property of their respective owners. This is not legal advice.
