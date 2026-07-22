from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProjectCommentEntry:
    project_name: str = ""
    on_model: float = 0
    laydown_detail: float = 0
    color_correct: float = 0
    post: float = 0
    model_hours: float = 0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Page1CommentsPayload:
    selected_internal_contact: dict[str, str]
    estimate_subject: str
    subtitle_line: str
    intro_text: str
    project_entries: tuple[ProjectCommentEntry, ...]
    custom_notes: str
    rendered_comments_block: str
    project_count: int
    project_count_label: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "selected_internal_contact": self.selected_internal_contact,
            "estimate_subject": self.estimate_subject,
            "subtitle_line": self.subtitle_line,
            "intro_text": self.intro_text,
            "project_entries": [entry.to_payload() for entry in self.project_entries],
            "custom_notes": self.custom_notes,
            "rendered_comments_block": self.rendered_comments_block,
            "project_count": self.project_count,
            "project_count_label": self.project_count_label,
        }


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _has_project_content(entry: ProjectCommentEntry) -> bool:
    return bool(entry.project_name) or any(
        value > 0
        for value in (
            entry.on_model,
            entry.laydown_detail,
            entry.color_correct,
            entry.post,
            entry.model_hours,
        )
    )


def normalize_project_entry(raw_entry: dict[str, Any]) -> ProjectCommentEntry:
    return ProjectCommentEntry(
        project_name=_clean_text(raw_entry.get("project_name")),
        on_model=_number(raw_entry.get("on_model")),
        laydown_detail=_number(raw_entry.get("laydown_detail")),
        color_correct=_number(raw_entry.get("color_correct")),
        post=_number(raw_entry.get("post")),
        model_hours=_number(raw_entry.get("model_hours")),
    )


def project_count_label(project_count: int) -> str:
    return "1 project=" if project_count == 1 else f"{project_count} projects="


def render_project_detail_line(entry: ProjectCommentEntry) -> str:
    parts = []
    if entry.on_model > 0:
        parts.append(f"On Model= {_format_number(entry.on_model)}")
    if entry.laydown_detail > 0:
        parts.append(f"Laydown/Detail={_format_number(entry.laydown_detail)}")
    if entry.color_correct > 0:
        parts.append(f"Color correct: {_format_number(entry.color_correct)}")
    if entry.post > 0:
        parts.append(f"Post= {_format_number(entry.post)}")
    if entry.model_hours > 0:
        parts.append(f"Model hrs= {_format_number(entry.model_hours)}")
    return ", ".join(parts)


def build_page1_comments_payload(
    *,
    selected_internal_contact: dict[str, str],
    estimate_subject: str,
    subtitle_line: str,
    project_entries: list[dict[str, Any]],
    custom_notes: str,
) -> Page1CommentsPayload:
    contact = dict(selected_internal_contact)
    subject = _clean_text(estimate_subject)
    subtitle = _clean_text(subtitle_line)
    notes = _clean_text(custom_notes)
    normalized_entries = tuple(
        entry for entry in (normalize_project_entry(raw) for raw in project_entries) if _has_project_content(entry)
    )
    count = len(normalized_entries)
    count_label = project_count_label(count)
    intro_text = f"Photography Estimate for {subject}:" if subject else "Photography Estimate:"

    lines = [
        f"Comments from {_clean_text(contact.get('name'))}",
        "",
        intro_text,
    ]
    if subtitle:
        lines.append(subtitle)
    lines.extend(["Estimate includes the following projects:", ""])

    for index, entry in enumerate(normalized_entries, start=1):
        lines.append(entry.project_name or f"Project {index}")
        detail_line = render_project_detail_line(entry)
        if detail_line:
            lines.append(detail_line)
        lines.append("")

    if notes:
        lines.append(notes)
        lines.append("")

    lines.append(count_label)

    return Page1CommentsPayload(
        selected_internal_contact=contact,
        estimate_subject=subject,
        subtitle_line=subtitle,
        intro_text=intro_text,
        project_entries=normalized_entries,
        custom_notes=notes,
        rendered_comments_block="\n".join(lines).strip(),
        project_count=count,
        project_count_label=count_label,
    )
