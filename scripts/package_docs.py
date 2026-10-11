"""
Copy the write-up, the three experiments (EXP-001, EXP-007, EXP-014) and the list
of all settings of the final model (docs/04) into this submission folder, together with every picture or file their links point to.

Files keep their place relative to the repository (outputs/..., src/...), so the
links in the copied .md files work inside the submission folder too. Links to
whole folders (e.g. "all step pictures") are not copied, to keep the folder small;
they are listed at the end so they can be pointed at the full repository.

Run (from the submission folder): python scripts/package_docs.py
"""
import re
import shutil
from pathlib import Path

SUB = Path(__file__).resolve().parent.parent
REPO = SUB.parent
DOCS = {"WRITEUP.md": "WRITEUP.md",
        "experiments/EXP-001_registration_feasibility.md": "experiments/EXP-001_registration_feasibility.md",
        "experiments/EXP-007_two_head_model.md": "experiments/EXP-007_two_head_model.md",
        "experiments/EXP-007_model_registry.md": "experiments/EXP-007_model_registry.md",
        "experiments/EXP-014_consistent_dataset.md": "experiments/EXP-014_consistent_dataset.md",
        "docs/04_final_model_parameters.md": "docs/04_final_model_parameters.md"}
LINK = re.compile(r"\]\(([^)#\s]+)\)")


def main():
    copied, folders, missing = 0, set(), set()
    docs = [SUB / d for d in DOCS.values()] + [SUB / "MODEL_COMPARISON.md"]
    for src, dst in DOCS.items():
        (SUB / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / src, SUB / dst)
    for doc in docs:
        for target in LINK.findall(doc.read_text()):
            if target.startswith("http"):
                continue
            in_repo = (REPO / doc.relative_to(SUB).parent / target).resolve()
            in_sub = (doc.parent / target).resolve()
            if in_repo.is_dir():
                folders.add(str(in_repo.relative_to(REPO)))
            elif in_repo.is_file():
                # copy when missing or when the repository has a newer version (a regenerated figure)
                if not in_sub.exists() or in_repo.stat().st_mtime > in_sub.stat().st_mtime:
                    in_sub.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(in_repo, in_sub)
                    copied += 1
            elif not in_sub.exists():
                missing.add(f"{doc.name}: {target}")
    print(f"copied {len(DOCS)} documents and {copied} linked files")
    print("folder links (not copied; see the full repository):", sorted(folders) or "none")
    print("broken links:", sorted(missing) or "none")


if __name__ == "__main__":
    main()
