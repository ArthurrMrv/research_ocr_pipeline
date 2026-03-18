from __future__ import annotations


class MissingPromptError(Exception):
    """Raised when a formatting step has no prompt for the given institution."""

    def __init__(self, step_name: str, institution: str | None) -> None:
        self.step_name = step_name
        self.institution = institution
        super().__init__(
            f"No prompt found for step '{step_name}' "
            f"(institution={institution!r})"
        )
