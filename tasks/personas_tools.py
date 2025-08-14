import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


PERSONAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "personas")
INDEX_CSV_PATH = os.path.join(PERSONAS_DIR, "personas_index.csv")
README_PATH = os.path.join(PERSONAS_DIR, "README_PERSONAS.md")


REQUIRED_FIELDS = [
    "Name:",
    "Age:",
    "Location:",
    "Profession:",
    "Backstory:",
    "Core Motivation:",
    "Fears & Insecurities:",
    "Hobbies & Passions:",
    "Media Diet:",
    "Communication Style:",
    "Quirk or Contradiction:",
    "Bio & Current Focus:",
]


REGION_MAP = {
    # Asia
    "Japan": ("Asia", "East Asia"),
    "India": ("Asia", "South Asia"),
    "Singapore": ("Asia", "Southeast Asia"),
    "Malaysia": ("Asia", "Southeast Asia"),
    "Indonesia": ("Asia", "Southeast Asia"),
    "Vietnam": ("Asia", "Southeast Asia"),
    "Taiwan": ("Asia", "East Asia"),
    "South Korea": ("Asia", "East Asia"),
    "Korea": ("Asia", "East Asia"),
    "China": ("Asia", "East Asia"),
    "Pakistan": ("Asia", "South Asia"),
    "Bangladesh": ("Asia", "South Asia"),
    "Oman": ("Asia", "Middle East"),
    "Israel": ("Asia", "Middle East"),
    # MENA / Africa
    "Egypt": ("Africa", "MENA"),
    "Morocco": ("Africa", "MENA"),
    "Nigeria": ("Africa", "West Africa"),
    "Ethiopia": ("Africa", "East Africa"),
    # Europe
    "France": ("Europe", "Western Europe"),
    "Germany": ("Europe", "Central Europe"),
    "Switzerland": ("Europe", "Western Europe"),
    "Italy": ("Europe", "Southern Europe"),
    "Portugal": ("Europe", "Southern Europe"),
    "Spain": ("Europe", "Southern Europe"),
    "Czechia": ("Europe", "Central Europe"),
    "Bulgaria": ("Europe", "Eastern Europe"),
    "Croatia": ("Europe", "Southeastern Europe"),
    "Poland": ("Europe", "Eastern Europe"),
    "Ukraine": ("Europe", "Eastern Europe"),
    "Ireland": ("Europe", "Western Europe"),
    "United Kingdom": ("Europe", "Western Europe"),
    "Scotland": ("Europe", "Western Europe"),
    # Americas
    "USA": ("Americas", "North America"),
    "United States": ("Americas", "North America"),
    "Canada": ("Americas", "North America"),
    "Brazil": ("Americas", "South America"),
    "Colombia": ("Americas", "South America"),
    "Argentina": ("Americas", "South America"),
    # Oceania
    "Australia": ("Oceania", "Australia"),
    "New Zealand": ("Oceania", "New Zealand"),
}


KEYWORD_TAGS = [
    (re.compile(r"\b(AI|ML|machine learning|model|MLOps|LLM|NLP|computer vision)\b", re.I), "AI/ML"),
    (re.compile(r"\b(NLP|language|linguistics)\b", re.I), "NLP"),
    (re.compile(r"\b(vision|camera|perception|CVPR|ADAS|driver assistance)\b", re.I), "Computer Vision"),
    (re.compile(r"\b(robot|robotics|autonom)\b", re.I), "Robotics"),
    (re.compile(r"\b(health|clinic|hospital|oncology|hospice|public health|pharma|diagnostic)\b", re.I), "Healthcare"),
    (re.compile(r"\b(education|teacher|curriculum|school|student|university)\b", re.I), "Education"),
    (re.compile(r"\b(energy|grid|solar|battery|power|utility)\b", re.I), "Energy"),
    (re.compile(r"\b(climate|flood|marine|ecolog|reef|conservation|air quality)\b", re.I), "Environment"),
    (re.compile(r"\b(transport|rail|mobility|bus|plow|driver|warehouse|logistics|supply chain)\b", re.I), "Mobility/Logistics"),
    (re.compile(r"\b(law|lawyer|legal|judge|court|immigration)\b", re.I), "Legal"),
    (re.compile(r"\b(fintech|finance|savings|insurer|payment|budget|product manager)\b", re.I), "Finance/Fintech"),
    (re.compile(r"\b(artisan|museum|art|cultural|design|conservator|photograph|music)\b", re.I), "Arts/Culture"),
    (re.compile(r"\b(agri|fish|fisher|farm|food|kitchen|nutrition)\b", re.I), "Food/Agriculture"),
    (re.compile(r"\b(accessib|disab|Deaf|sign language)\b", re.I), "Accessibility"),
    (re.compile(r"\b(security|cyber|incident response|fraud)\b", re.I), "Security/Risk"),
    (re.compile(r"\b(manufactur|machinist|upholster|aerospace|shop|CNC)\b", re.I), "Manufacturing"),
    (re.compile(r"\b(government|municipal|policy|public)\b", re.I), "Public Sector"),
]


def parse_persona(filepath: str) -> Tuple[Dict[str, str], List[str]]:
    """Parse fields and return (fields, missing_fields)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    fields: Dict[str, str] = {}
    missing: List[str] = []

    # Simple field extraction for the first four lines
    lines = content.splitlines()
    for key in ["Name:", "Age:", "Location:", "Profession:"]:
        val = None
        for line in lines[:12]:
            if line.startswith(key):
                val = line.split(":", 1)[1].strip()
                break
        if val is None:
            missing.append(key)
            fields[key[:-1]] = ""
        else:
            fields[key[:-1]] = val

    # Section presence validation
    for section in REQUIRED_FIELDS:
        if section not in content:
            if section not in missing:
                missing.append(section)

    return fields, missing


def derive_region_tags(location_value: str) -> List[str]:
    if not location_value:
        return []
    parts = [p.strip() for p in location_value.split(",")]
    country = parts[-1] if parts else location_value
    region_tags: List[str] = []
    # Normalize some country names embedded in cities
    country_clean = country
    # Special-case for United Kingdom constituents
    if country_clean in {"Scotland"}:
        country_clean = "United Kingdom"
    if country_clean in REGION_MAP:
        primary, secondary = REGION_MAP[country_clean]
        region_tags.extend([primary, secondary])
        if primary == "Africa" and secondary == "MENA":
            region_tags.append("MENA")
        if primary == "Asia" and secondary == "Middle East":
            region_tags.append("MENA")
    return region_tags


def derive_domain_tags(profession_value: str) -> List[str]:
    tags: List[str] = []
    for pattern, tag in KEYWORD_TAGS:
        if pattern.search(profession_value or ""):
            tags.append(tag)
    return sorted(set(tags))


def build_index(personas_dir: str) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
    rows: List[Dict[str, str]] = []
    validation: Dict[str, List[str]] = {}
    for fname in sorted(os.listdir(personas_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(personas_dir, fname)
        fields, missing = parse_persona(fpath)
        if missing:
            validation[fname] = missing
        name = fields.get("Name", "")
        age = fields.get("Age", "")
        location = fields.get("Location", "")
        profession = fields.get("Profession", "")
        tags = sorted(set(derive_region_tags(location) + derive_domain_tags(profession)))
        rows.append({
            "file": fname,
            "name": name,
            "age": age,
            "location": location,
            "profession": profession,
            "tags": ";".join(tags),
        })
    return rows, validation


def write_csv(rows: List[Dict[str, str]], csv_path: str) -> None:
    fieldnames = ["file", "name", "age", "location", "profession", "tags"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: List[Dict[str, str]]) -> Tuple[Counter, Counter]:
    region_counter: Counter = Counter()
    domain_counter: Counter = Counter()
    for row in rows:
        tags = (row.get("tags") or "").split(";") if row.get("tags") else []
        for t in tags:
            if t in {"Asia", "Europe", "Africa", "Americas", "Oceania", "MENA", "East Asia", "South Asia", "Southeast Asia", "Middle East", "North America", "South America", "Western Europe", "Eastern Europe", "Central Europe", "Southeastern Europe", "Australia", "New Zealand"}:
                region_counter[t] += 1
            else:
                if t:
                    domain_counter[t] += 1
    return region_counter, domain_counter


def write_readme(rows: List[Dict[str, str]], out_path: str) -> None:
    region_counter, domain_counter = summarize(rows)
    total = len(rows)

    def fmt_counter(c: Counter) -> str:
        parts = [f"- {k}: {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
        return "\n".join(parts) if parts else "- (none)"

    content = []
    content.append("## Personas Dataset\n")
    content.append(f"Total personas: {total}\n")
    content.append("\n")
    content.append("### Quick Index\n")
    content.append("The file `personas_index.csv` contains: `file`, `name`, `age`, `location`, `profession`, `tags`.\n")
    content.append("Tags are inferred heuristically from location and profession keywords (e.g., AI/ML, Healthcare, Environment, MENA, East Asia).\n\n")
    content.append("### Coverage\n")
    content.append("Regions (approximate):\n")
    content.append(fmt_counter(region_counter) + "\n\n")
    content.append("Domains (keyword-derived):\n")
    content.append(fmt_counter(domain_counter) + "\n\n")
    content.append("### Structure\n")
    content.append("Each persona `.txt` follows this schema (all fields present):\n")
    content.append("- Name\n- Age\n- Location\n- Profession\n- Backstory\n- Core Motivation\n- Fears & Insecurities\n- Hobbies & Passions\n- Media Diet\n- Communication Style\n- Quirk or Contradiction\n- Bio & Current Focus\n\n")
    content.append("### Suggested Use Cases\n")
    content.append("- Product/user research exercises: recruit diverse personas by tag or region\n")
    content.append("- Prompting/eval datasets for UX writing and AI assistants (grounded, realistic profiles)\n")
    content.append("- Storytelling and scenario planning across industries (health, mobility, public sector)\n")
    content.append("- Teaching materials for ethics, data governance, and inclusive design\n\n")
    content.append("### Validation\n")
    content.append("Use the validation mode of `personas_tools.py` to check for missing fields.\n")
    content.append("Example: `python deckoviz_screening_test/tasks/personas_tools.py --validate`\n\n")
    content.append("### Notes\n")
    content.append("- Tags are best-effort. Adjust `KEYWORD_TAGS` and `REGION_MAP` in `personas_tools.py` for your context.\n")
    content.append("- CSV excludes the long narrative fields to keep the index compact; see individual files for full detail.\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(content))


def main():
    parser = argparse.ArgumentParser(description="Build and validate personas index.")
    parser.add_argument("--personas-dir", default=PERSONAS_DIR, help="Directory with persona .txt files")
    parser.add_argument("--write-index", action="store_true", help="Write personas_index.csv")
    parser.add_argument("--validate", action="store_true", help="Validate required fields and print report")
    parser.add_argument("--write-readme", action="store_true", help="Write README_PERSONAS.md summary")
    args = parser.parse_args()

    rows, validation = build_index(args.personas_dir)

    if args.validate:
        if not validation:
            print("Validation: OK (all required fields present)")
        else:
            print("Validation: MISSING FIELDS detected in the following files:")
            for fname, missing in sorted(validation.items()):
                print(f"- {fname}: {', '.join(missing)}")

    if args.write_index:
        write_csv(rows, INDEX_CSV_PATH)
        print(f"Wrote index: {INDEX_CSV_PATH}")

    if args.write_readme:
        write_readme(rows, README_PATH)
        print(f"Wrote README: {README_PATH}")


if __name__ == "__main__":
    main()


