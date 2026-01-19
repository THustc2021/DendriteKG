from __future__ import annotations

import os, json
import tqdm

from tools.llm import create_chat_model

with open("config/api_config.json", "r") as f:
    config = json.load(f)
worker_llm = create_chat_model(config)

task_description1 = """
<Task>
You are an expert scientific modeler in electrochemistry and phase-field methods.

I will provide an abstract or paper text about lithium-ion batteries and/or dendrite growth. The paper may be experimental, theoretical, or computational, and may or may not use a phase-field model.

GOAL:
Extract and synthesize ALL information that can help design a physically reasonable phase-field model for lithium dendrite growth in a lithium-ion battery with a LIQUID electrolyte.

CRITICAL RULES:
1) Be comprehensive: include any mechanism, condition, variable, relation, parameter, constraint, or modeling choice relevant to dendrites and electrodeposition.
2) Do NOT invent facts. If something is not explicitly stated, either omit it or label it as "Not stated".
3) Always attach evidence: for every extracted claim, include a short evidence snippet from the provided text (5–30 words) OR a very local paraphrase that is clearly attributable.
4) Separate "Directly stated" vs "Implied/Interpretive" insights. Interpretive insights must be clearly marked and tied to evidence.
5) If the work does not use phase-field, still extract modeling ingredients and map them to possible phase-field components (free energy terms, evolution equations, couplings, BCs) but keep this mapping explicitly labeled as a “Modeling mapping suggestion”.

OUTPUT:
Return the following sections exactly in this order. Use concise bullets. Do not add extra sections.

[0) Paper ID]
- Title:
- Year (if stated):
- DOI (if stated):

[1) Material & System Definition]
Extract what system is studied.
- Electrode/anode material(s):
- Counter/reference electrode (if stated):
- Electrolyte (liquid) composition / salt / solvent / concentration (if stated):
- Separator/porous medium (if stated):
- Interfaces explicitly discussed (Li/electrolyte, SEI, etc.):
Evidence:
- ...

[2) Operating / Physical Conditions]
Extract boundary/operating conditions and control modes.
- Control mode(s): (constant current / constant voltage / pulsed / galvanostatic / potentiostatic / other)
- Temperature:
- Current density / C-rate / overpotential:
- Pressure / stack pressure / mechanical constraint:
- Geometry / dimensionality (1D/2D/3D), characteristic length (if stated):
Evidence:
- ...

[3) Observed or Modeled Phenomena]
What was observed/measured/simulated?
- Dendrite-related phenomena (nucleation, tip growth, mossy Li, instability, short-circuit, etc.):
- Spatiotemporal descriptors (where/when, morphology descriptors, growth direction, roughness):
- Key outcomes (thresholds, regimes, maps):
Evidence:
- ...

[4) Mechanisms & Causal Statements]
Extract mechanism-level statements (transport, kinetics, thermodynamics, mechanics, interfacial effects).
For each mechanism, use the template:
- Mechanism:
  - What it affects:
  - Direction/trend (increase/decrease/promotes/suppresses) if stated:
  - Evidence:
Repeat for all mechanisms mentioned.

[5) Variables, Fields, and Measured/Modeled Quantities]
List all explicit variables/fields used or discussed.
Examples: ion concentration, electric potential, current density, overpotential, exchange current, SEI thickness, surface energy, curvature, diffusivity, transference number, conductivity, etc.
Format:
- Variable/field: definition/context (if stated) | units (if stated) | evidence

[6) Explicit Physical Relationships / Governing Laws]
Extract explicit relationships, equations, or named laws.
Include:
- Transport: Nernst–Planck, diffusion, migration, electroneutrality, Poisson, porous media relations
- Kinetics: Butler–Volmer, Tafel, Marcus (if mentioned)
- Interfacial thermodynamics: Gibbs–Thomson/curvature, surface energy, nucleation relations
- Mechanics: stress effects, pressure-dependent deposition, fracture (if mentioned)
For each:
- Relationship (equation name or text description):
- Variables involved:
- Applicable conditions/assumptions (if stated):
- Evidence:

[7) Parameters and Numerical Values]
Extract any parameters and their values/ranges.
Format each as:
- Parameter: value/range | how obtained (fit/assumed/measured) | evidence

[8) If a Phase-Field Model Is Used (ONLY if explicitly used)]
If the paper explicitly uses phase-field, extract the model details:
- Order parameter(s) and meaning:
- Free energy functional terms (bulk, gradient/interfacial, electrochemical, elastic, etc.):
- Evolution equation type (Allen–Cahn / Cahn–Hilliard / other) and coupling fields:
- Treatment of electrochemical reactions at the interface:
- Boundary/initial conditions:
- Numerical setup (mesh, timestep, solver) only if stated:
Evidence:
- ...

[9) Modeling Mapping Suggestions Toward a Phase-Field Dendrite Model (clearly labeled)]
Even if no phase-field is used, map extracted knowledge into phase-field model “ingredients”.
Use this structured list:
- Suggested phase-field ingredient:
  - Source mechanism/relationship from sections [4–6]:
  - How it could enter a phase-field framework (e.g., free energy term, mobility, source term, coupling, BC):
  - Confidence: High/Medium/Low
  - Evidence pointer: (cite the evidence snippets above)

[10) Scope Limits / Missing Pieces]
- What the paper explicitly does NOT cover (if stated):
- Key limitations/assumptions mentioned:
Evidence:
- ...

Remember: be comprehensive and evidence-grounded. 
</Task>
"""

task_description2 = """
<Task>
You are an information extraction system for building a reliable knowledge graph.

INPUT:
You will be given a paper summary text (generated in a previous step). The summary may contain sections such as:
Material/System, Conditions, Phenomena, Mechanisms, Variables, Relationships/Laws, Parameters, and Phase-field mapping notes.

GOAL:
Extract comprehensive knowledge graph triples to support designing a physically reasonable PHASE-FIELD model of lithium dendrite growth in lithium-ion batteries with LIQUID electrolytes.

SCHEMA (YOU MUST FOLLOW):
Entity types: Paper, Material, Electrolyte, Interface, CellComponent, Geometry, Condition, Quantity, Parameter, Phenomenon, Mechanism, Relationship, Model, ModelComponent, PhaseFieldIngredient.

Allowed predicates:
STUDIES, REPORTS, USES_MODEL,
OCCURS_UNDER, TRIGGERED_UNDER, INVOLVES,
LEADS_TO, AFFECTS,
HAS_COMPONENT, USES_RELATIONSHIP, MAPS_TO,
PARAMETER_OF.

STRICT RULES:
1) Only create a triple if it is explicitly supported by the input summary text OR if it is explicitly labeled in the summary as a “mapping suggestion”.
2) Every triple MUST include an evidence snippet copied from the input (5–40 words). No evidence = no triple.
3) Do not invent entity names, parameter values, or relationships.
4) For interpretive “MAPS_TO” triples, set certainty="interpretive" and provide confidence (high/medium/low). All other triples should be certainty="explicit".
5) Normalize entity labels lightly:
   - Use canonical names if the text provides them (e.g., “Butler–Volmer”, “Nernst–Planck”).
   - Otherwise keep the original phrasing.
6) IDs:
   - paper_id: use the DOI if present; else use a stable placeholder from the summary.
   - entity_id: create stable IDs by type + short_name (snake_case). Example: "Mechanism_concentration_polarization".

OUTPUT FORMAT (JSON Lines, one triple per line):
{
  "subject_id": "...",
  "subject_type": "...",
  "predicate": "...",
  "object_id": "...",
  "object_type": "...",
  "paper_id": "...",
  "evidence": "...",
  "certainty": "explicit|interpretive",
  "confidence": "high|medium|low|null",
  "qualifiers": {
     "conditions": ["..."], 
     "values": [{"name":"...", "value":"...", "unit":"..."}]
  }
}

GUIDANCE:
- Always extract:
  (Paper) STUDIES (Material/Electrolyte/Interface)
  (Paper) REPORTS (Phenomenon/Mechanism/Relationship/Parameter)
  (Phenomenon) OCCURS_UNDER (Condition) when conditions are present
  (Mechanism/Phenomenon) INVOLVES (Quantity) when variables are present
  (Model) HAS_COMPONENT (ModelComponent) if any model details appear
- If phase-field is used, also extract PhaseFieldIngredient triples.
- If the summary mentions a mapping suggestion, encode it as MAPS_TO.
</Task>
"""

# resource_dir = "output/scopus_dendrite"
resource_dir = "output/step1_results"
output_dir = "output/step2_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for od in os.listdir(resource_dir):

    output_sub_dir = os.path.join(output_dir, od)
    if not os.path.exists(output_sub_dir):
        os.mkdir(output_sub_dir)

    for f in tqdm.tqdm(os.listdir(os.path.join(resource_dir, od))):

        if os.path.exists(os.path.join(output_sub_dir, f[:-4] + ".txt")):
            continue

        file_p = os.path.join(resource_dir, od, f)
        # content = extract_elsevier_fulltext_xml(file_p)['full_text']
        with open(file_p, "r", encoding="utf-8") as file:
            content = file.read()

        task_formatted = task_description2 + f"""
<Content>
{content}
</Content>
"""

        message = worker_llm.invoke(task_formatted)
        print(message.content)

        with open(os.path.join(output_sub_dir, f[:-4] + ".txt"), "w") as file:
            file.write(message.content)

# input tokens 6052, output tokens 7400
# 9119, 1980
# 8988, 1632
# 6114, 1297
# 8335, 1337