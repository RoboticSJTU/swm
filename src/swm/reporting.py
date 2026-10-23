"""Shared Markdown layout for compact evaluation reports."""
from html import escape


def markdown_table(headers, rows):
    def line(values):
        return "| " + " | ".join(str(value).replace("|", r"\|").replace("\n", " ") for value in values) + " |"
    return [line(headers), "| --- |" + " ---: |" * (len(headers) - 1), *(line(row) for row in rows)]


def markdown_details(title, body):
    return ["<details>", f"<summary>{escape(str(title))}</summary>", "", *body, "</details>", ""]
