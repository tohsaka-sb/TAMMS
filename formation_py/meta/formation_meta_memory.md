# Formation Meta Memory

## General Principles

- Use WorkspaceBrief, not full file content.
- Choose one primary intent and one or more secondary intents.
- Do not invent operators.
- Prefer document selectors over all-document processing.
- Output valid JSON only when asked for a plan.
- The same operator may be used multiple times only when selectors and purposes differ.

## Intent Strategies

- overview: use GistOperator, CAMClusterOperator, ClusterFusionOperator.
- debug: use EvidencePreservingCompressor, SemanticFactExtractor, LinkEvolutionOperator; add ProceduralExtractor if build/run files exist.
- reproduce: use ProceduralExtractor and EvidencePreservingCompressor over documentation, build scripts and run scripts.
- locate: use GistOperator, SemanticFactExtractor, LinkEvolutionOperator.
- graph: use AMemNoteOperator, SemanticFactExtractor, LinkEvolutionOperator.
- summarize: use GistOperator, CAMClusterOperator, ClusterFusionOperator.

## Operator Guidelines

- The same operator may be used multiple times when selectors and purposes differ.
- Do not repeat the same operator with identical selector and purpose.
- Give every step a unique step_id.

## Selector Guidelines

- Prefer document selectors over all-document processing.
- If a doc_id is missing, do not guess; use path/content_roles selectors instead.

## Successful Patterns

## Failure Lessons

- If a selector returns zero documents, revise selector using roles and path patterns from WorkspaceBrief.
- Do not request complete file contents during planning.

## Anti-Patterns

- Do not repeat the same operator with identical selector and purpose.

## Update History
