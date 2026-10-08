from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from servemind.config.settings import SKILLS_ROOT


HIGH_RISK_INTENTS = {"refund_policy", "cancel_policy", "complaint", "human_handoff"}
ROLE_PERMISSIONS = {
    "buyer": frozenset({"catalog_read", "policy_public"}),
    "merchant": frozenset({"merchant_read", "policy_public"}),
}


@dataclass(frozen=True)
class Skill:
    name: str
    audience: str
    keywords: tuple[str, ...]
    content: str
    intents: tuple[str, ...] = ("*",)
    priority: int = 0
    required_permissions: tuple[str, ...] = ()
    valid_from: date | None = None
    valid_until: date | None = None
    conflict_group: str = ""
    source: str = ""
    denied_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillSelection:
    skills: tuple[Skill, ...]
    conflicts: tuple[str, ...]
    expired: tuple[str, ...]


class SkillManager:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or SKILLS_ROOT
        self.skills: list[Skill] = []
        self.load()

    @staticmethod
    def _list(value: str) -> tuple[str, ...]:
        return tuple(item.strip() for item in value.split(",") if item.strip())

    def load(self) -> None:
        self.skills = []
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            if text.startswith("---"):
                _, _, rest = text.partition("---\n")
                header, _, body = rest.partition("\n---")
            else:
                header, body = "", text
            metadata: dict[str, str] = {}
            for line in header.replace("---", "").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
            start = date.fromisoformat(metadata["valid_from"]) if metadata.get("valid_from") else None
            end = date.fromisoformat(metadata["valid_until"]) if metadata.get("valid_until") else None
            if start and end and end < start:
                raise ValueError(f"invalid Skill validity: {path}")
            self.skills.append(Skill(
                name=metadata.get("name", path.parent.name),
                audience=metadata.get("audience", "buyer"),
                keywords=self._list(metadata.get("keywords", "")),
                content=body.strip() or text,
                intents=self._list(metadata.get("intents", "*")),
                priority=int(metadata.get("priority", "0")),
                required_permissions=self._list(metadata.get("required_permissions", "")),
                valid_from=start,
                valid_until=end,
                conflict_group=metadata.get("conflict_group", ""),
                source=str(path),
                denied_tools=self._list(metadata.get("denied_tools", "")),
            ))

    def select(self, message: str, intent: str, actor_type: str = "buyer",
               *, granted_permissions: set[str] | None = None,
               as_of: date | None = None) -> SkillSelection:
        granted = (set(ROLE_PERMISSIONS.get(actor_type, frozenset()))
                   if granted_permissions is None else granted_permissions)
        today = as_of or date.today()
        text = (message or "").lower()
        candidates: list[tuple[int, int, Skill]] = []
        expired: list[str] = []
        for skill in self.skills:
            if skill.audience not in (actor_type, "both", ""):
                continue
            if not set(skill.required_permissions).issubset(granted):
                continue
            if "*" not in skill.intents and intent not in skill.intents:
                continue
            hits = sum(keyword.lower() in text for keyword in skill.keywords)
            if not hits and ("*" in skill.intents or intent not in HIGH_RISK_INTENTS):
                continue
            if (skill.valid_from and today < skill.valid_from) or (skill.valid_until and today > skill.valid_until):
                expired.append(skill.name)
                continue
            candidates.append((skill.priority, hits, skill))
        candidates.sort(key=lambda row: (-row[0], -row[1], row[2].name))
        chosen: list[Skill] = []
        groups: dict[str, tuple[int, Skill]] = {}
        conflicts: list[str] = []
        for priority, _hits, skill in candidates:
            group = skill.conflict_group
            if not group:
                chosen.append(skill)
                continue
            if group in conflicts:
                continue
            previous = groups.get(group)
            if previous is None:
                groups[group] = (priority, skill)
            elif previous[0] == priority:
                conflicts.append(group)
                groups.pop(group)
            # Lower-priority conflicting Skills are intentionally suppressed.
        chosen.extend(skill for group, (_priority, skill) in groups.items() if group not in conflicts)
        chosen.sort(key=lambda skill: (-skill.priority, skill.name))
        return SkillSelection(tuple(chosen[:3]), tuple(sorted(set(conflicts))),
                              tuple(sorted(set(expired))))

    def relevant(self, message: str, actor_type: str = "buyer") -> list[Skill]:
        return list(self.select(message, "other", actor_type).skills)

    def relevant_for_intent(self, message: str, intent: str,
                            actor_type: str = "buyer") -> list[Skill]:
        return list(self.select(message, intent, actor_type).skills)

    def prompt(self, message: str, actor_type: str = "buyer") -> str:
        return "\n\n".join(f"[Skill: {skill.name}]\n{skill.content[:5000]}"
                           for skill in self.relevant(message, actor_type))

    def prompt_for_intent(self, message: str, intent: str,
                          actor_type: str = "buyer") -> str:
        return "\n\n".join(f"[Skill: {skill.name}]\n{skill.content[:5000]}"
                           for skill in self.relevant_for_intent(message, intent, actor_type))

    def summary(self) -> dict:
        return {"count": len(self.skills), "skills": [
            {"name": skill.name, "audience": skill.audience,
             "keywords": list(skill.keywords), "intents": list(skill.intents),
             "priority": skill.priority, "required_permissions": list(skill.required_permissions),
             "valid_from": skill.valid_from.isoformat() if skill.valid_from else None,
             "valid_until": skill.valid_until.isoformat() if skill.valid_until else None}
            for skill in self.skills]}
