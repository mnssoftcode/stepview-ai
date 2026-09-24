You are the lead technical project-documentation AI for a new computer-vision project called "StepView AI".

Before writing any implementation code, create a complete project documentation package that becomes the single source of truth for all future AI coding sessions.

Project idea:
A camera observes terrain in front of a walking/hiking person. AI identifies visually suitable candidate regions for foot placement and overlays recommendations. The system must communicate uncertainty and must never claim guaranteed physical safety.

The long-term target is an on-device mobile application.

Development strategy:
1. Train/evaluate models in Python/PyTorch.
2. Export the inference model to ONNX.
3. Build a React web prototype that runs inference in the browser.
4. Validate the computer-vision pipeline.
5. Reuse the model for a React Native mobile prototype.
6. Optimize for real-time on-device inference.

Create these files:

1. docs/00_PROJECT_MASTER.md
   - project goal
   - scope
   - non-goals
   - development phases
   - success criteria
   - major research questions

2. docs/01_PRD.md
   - target user
   - problem
   - user flow
   - MVP
   - UI requirements
   - acceptance criteria
   - safety language

3. docs/02_ML_SPEC.md
   - ML problem definition
   - model decomposition
   - classes
   - inputs/outputs
   - candidate foot-placement logic
   - uncertainty
   - training strategy
   - evaluation metrics
   - data split strategy
   - experiment tracking

4. docs/03_DATASET_AND_ANNOTATION.md
   - data sources
   - collection protocol
   - annotation schema
   - difficult examples
   - hard negatives
   - dataset quality checks
   - dataset versioning

5. docs/04_SYSTEM_ARCHITECTURE.md
   - training architecture
   - web inference architecture
   - mobile architecture
   - model format
   - repository structure
   - separation of ML/UI/inference

6. docs/05_DEVELOPMENT_ROADMAP.md
   - phase-by-phase tasks
   - deliverables
   - exit criteria
   - field testing
   - optimization
   - portfolio release

7. docs/06_AI_AGENT_RULES.md
   - how a future coding AI must work
   - mandatory documents to read
   - no scope creep
   - experiment discipline
   - definition of done
   - required reporting format

Important:
- Do not claim the system can guarantee safety.
- Do not add unrelated features.
- Favor experiments and measurable validation.
- Keep the architecture compatible with both browser and mobile inference.
- Make decisions explicit and reversible where possible.
