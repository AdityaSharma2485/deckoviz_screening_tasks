## Personas Dataset
Total personas: 50

### Quick Index
The file `personas_index.csv` contains: `file`, `name`, `age`, `location`, `profession`, `tags`.
Tags are inferred heuristically from location and profession keywords (e.g., AI/ML, Healthcare, Environment, MENA, East Asia).

### Coverage
Regions (approximate):
- Asia: 17
- Europe: 12
- Americas: 8
- Africa: 5
- East Asia: 5
- MENA: 5
- North America: 5
- South Asia: 5
- Southeast Asia: 5
- Western Europe: 5
- South America: 3
- Central Europe: 2
- Eastern Europe: 2
- Middle East: 2
- Oceania: 2
- Australia: 1
- New Zealand: 1
- Southeastern Europe: 1

Domains (keyword-derived):
- Healthcare: 8
- Education: 6
- Environment: 6
- Mobility/Logistics: 6
- Public Sector: 6
- AI/ML: 5
- Arts/Culture: 4
- Energy: 4
- Finance/Fintech: 3
- Manufacturing: 3
- NLP: 3
- Security/Risk: 3
- Robotics: 2
- Southern Europe: 2
- Accessibility: 1
- Computer Vision: 1
- East Africa: 1
- Food/Agriculture: 1
- Legal: 1
- West Africa: 1

### Structure
Each persona `.txt` follows this schema (all fields present):
- Name
- Age
- Location
- Profession
- Backstory
- Core Motivation
- Fears & Insecurities
- Hobbies & Passions
- Media Diet
- Communication Style
- Quirk or Contradiction
- Bio & Current Focus

### Suggested Use Cases
- Product/user research exercises: recruit diverse personas by tag or region
- Prompting/eval datasets for UX writing and AI assistants (grounded, realistic profiles)
- Storytelling and scenario planning across industries (health, mobility, public sector)
- Teaching materials for ethics, data governance, and inclusive design

### Validation
Use the validation mode of `personas_tools.py` to check for missing fields.
Example: `python deckoviz_screening_test/tasks/personas_tools.py --validate`

### Notes
- Tags are best-effort. Adjust `KEYWORD_TAGS` and `REGION_MAP` in `personas_tools.py` for your context.
- CSV excludes the long narrative fields to keep the index compact; see individual files for full detail.
