"""Clean game boards with deterministic artwork variation and state projections."""

from __future__ import annotations
import hashlib
import math
import re
from collections import defaultdict
from functools import lru_cache
from types import SimpleNamespace
from PIL import Image, ImageDraw, ImageFont, ImageChops
from .model import parse_sexpr, section

WIDTH, HEIGHT = 1440, 1080
VERSION = "clean-scenes-v2"
STYLES = ("classic", "pastel", "pixel", "night")
COLORS = ["#d56a42", "#308cae", "#dfb036", "#73a351", "#a969ac", "#496fba"]
INK = "#263442"
PALETTES = {
    "classic": {
        "floor": "#ede5ce",
        "wall": "#927c59",
        "ice": "#bbdfe8",
        "water": "#234e73",
        "ground": "#faf7ee",
        "line": "#baa88c",
    },
    "pastel": {
        "floor": "#f6e4de",
        "wall": "#b7abc9",
        "ice": "#d1eef2",
        "water": "#718eae",
        "ground": "#fcf8fb",
        "line": "#c5b9c7",
    },
    "pixel": {
        "floor": "#dceab3",
        "wall": "#657c5d",
        "ice": "#b4e6e3",
        "water": "#2e7191",
        "ground": "#f3f5df",
        "line": "#9cba87",
    },
    "night": {
        "floor": "#adc7d2",
        "wall": "#1d304d",
        "ice": "#93c9da",
        "water": "#193c62",
        "ground": "#dae5e8",
        "line": "#718e9a",
    },
}


@lru_cache(maxsize=96)
def font(size, bold=False):
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans"
        + ("-Bold" if bold else "")
        + ".ttf",
        size,
    )


def color(name):
    return COLORS[int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % len(COLORS)]


def tint(value, delta):
    value = value.lstrip("#")
    return "#" + "".join(
        f"{max(0, min(255, int(value[i : i + 2], 16) + delta)):02x}" for i in (0, 2, 4)
    )


def stacks(state, roots, predicate="on"):
    above = {b: a for a, b in state[predicate]}
    result = []
    for base in roots:
        pile = []
        seen = set()
        current = base
        while current in above:
            current = above[current]
            if current in seen:
                raise ValueError("Cyclic stack")
            seen.add(current)
            pile.append(current)
        result.append(pile)
    return result


class Scene:
    def __init__(self, scenario, style=None):
        self.game = scenario["game"]
        self.domain = scenario["domain"]
        self.style = STYLES[
            int(
                hashlib.sha256(
                    (self.game + ":" + str(scenario["seed"]) + ":art").encode()
                ).hexdigest()[:8],
                16,
            )
            % len(STYLES)
        ]
        self.style = style or self.style
        self.palette = PALETTES[self.style]
        self.initial = section(parse_sexpr(scenario["problem"]), ":init")[1:]
        self.state = defaultdict(list)
        for a in self.initial:
            self.state[a[0]].append(a[1:])
        self.facts = {tuple(a) for a in self.initial}
        self.image = Image.new("RGB", (WIDTH, HEIGHT), self.palette["ground"])
        self.draw = ImageDraw.Draw(self.image)
        self.entities = []
        self.labels = []
        self.represented = set()
        self.details = []
        self.regions = {}

    def label(self, xy, value, size=23, bold=False, fill=INK, anchor="left"):
        value = str(value)
        x, y = xy
        box = self.draw.textbbox((0, 0), value, font=font(size, bold))
        w, h = box[2] - box[0], box[3] - box[1]
        if anchor == "center":
            x -= w / 2
        elif anchor == "right":
            x -= w
        bbox = [round(x), round(y), round(x + w), round(y + h)]
        if min(bbox[:2]) < 0 or bbox[2] > WIDTH or bbox[3] > HEIGHT:
            raise ValueError((self.game, "Clipped label", value, bbox))
        self.draw.text((x, y - box[1]), value, font=font(size, bold), fill=fill)
        self.labels.append({"text": value, "bbox": bbox, "font_size": size})
        return bbox

    def texture(self, box, kind):
        fill = {
            "ice": self.palette["ice"],
            "water": self.palette["water"],
            "brick": self.palette["wall"],
            "stone": self.palette["floor"],
            "tile": self.palette["floor"],
            "wood": self.palette["floor"],
            "darkwood": self.palette["ground"],
            "sand": self.palette["ground"],
            "felt": {
                "classic": "#397463",
                "pastel": "#936780",
                "pixel": "#286e79",
                "night": "#29445e",
            }[self.style],
            "metal": "#abbfc8",
        }[kind]
        self.draw.rectangle(tuple(box), fill=fill)
        if self.style == "classic" and kind in {"wood", "brick"}:
            x0, y0, x1, y1 = map(int, box)
            for y in range(y0 + 14, y1, 26):
                self.draw.line((x0, y, x1, y), fill=tint(fill, -14), width=1)
        if self.style == "pixel" and kind == "brick":
            x0, y0, x1, y1 = map(int, box)
            for y in range(y0, y1, 32):
                self.draw.line((x0, y, x1, y), fill=tint(fill, -22), width=2)
                for x in range(x0 + (16 if (y - y0) // 32 % 2 else 0), x1, 40):
                    self.draw.line(
                        (x, y, x, min(y + 32, y1)), fill=tint(fill, -22), width=2
                    )

    def fact(self, predicate, *args):
        atom = (predicate, *args)
        if atom not in self.facts:
            raise ValueError((self.game, "Absent visual fact", atom))
        self.represented.add(atom)

    def present(self, predicate, *args):
        return (predicate, *args) in self.facts

    def note(self, xy, value, atoms=(), size=21, fill=INK):
        bbox = self.label(xy, value, size, False, fill)
        for a in atoms:
            self.fact(*a)
        self.details.append(
            {"text": value, "bbox": bbox, "facts": [list(a) for a in atoms]}
        )

    def object(
        self, name, kind, center, size=72, location=None, attributes=None, label=True
    ):
        x, y = center
        accent = color(name)
        if self.style == "pastel":
            accent = tint(accent, 35)
        stamp = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        sprite(SimpleNamespace(draw=ImageDraw.Draw(stamp)), kind, (80, 80), 115, accent)
        if self.style == "pixel":
            stamp = stamp.resize((24, 24), Image.Resampling.NEAREST).resize(
                (160, 160), Image.Resampling.NEAREST
            )
        elif self.style == "night":
            stamp = stamp.quantize(colors=10).convert("RGBA")
        stamp = stamp.resize(
            (round(size * 1.35), round(size * 1.35)),
            Image.Resampling.NEAREST
            if self.style == "pixel"
            else Image.Resampling.LANCZOS,
        )
        self.image.paste(
            stamp, (round(x - stamp.width / 2), round(y - stamp.height / 2)), stamp
        )
        bbox = [
            round(x - size * 0.6),
            round(y - size * 0.6),
            round(x + size * 0.6),
            round(y + size * 0.6),
        ]
        if min(bbox[:2]) < 0 or bbox[2] > WIDTH or bbox[3] > HEIGHT:
            raise ValueError((self.game, "Clipped entity", name, bbox))
        if label:
            self.label(
                (x, y + size * 0.6 + 3),
                name,
                22 if len(name) < 15 else 17,
                True,
                anchor="center",
            )
        self.entities.append(
            {
                "id": name,
                "kind": kind,
                "bbox": bbox,
                "location": location,
                "attributes": attributes or {},
            }
        )
        return bbox

    def arrow(self, a, b, fill="#315d72", width=5, dashed=False, head=True):
        self.draw.line((*a, *b), fill=fill, width=width)
        if head:
            ax, ay = a
            bx, by = b
            angle = math.atan2(by - ay, bx - ax)
            self.draw.polygon(
                [
                    b,
                    (
                        bx - 16 * math.cos(angle - 0.45),
                        by - 16 * math.sin(angle - 0.45),
                    ),
                    (
                        bx - 16 * math.cos(angle + 0.45),
                        by - 16 * math.sin(angle + 0.45),
                    ),
                ],
                fill=fill,
            )

    def finish(self):
        missing = {
            a for a in self.facts if a[0] in dynamic_predicates(self.domain)
        } - self.represented
        if missing:
            raise ValueError((self.game, "Undrawn dynamic state", sorted(missing)))
        background = Image.new("RGB", self.image.size, self.palette["ground"])
        bounds = ImageChops.difference(self.image, background).getbbox()
        background.close()
        x0, y0, x1, y1 = bounds
        x0, y0 = max(0, x0 - 24), max(0, y0 - 24)
        x1, y1 = min(WIDTH, x1 + 24), min(HEIGHT, y1 + 24)
        cropped = self.image.crop((x0, y0, x1, y1))
        self.image.close()
        self.image = cropped
        for item in self.entities + self.labels + self.details:
            bx0, by0, bx1, by1 = item["bbox"]
            item["bbox"] = [bx0 - x0, by0 - y0, bx1 - x0, by1 - y0]
        self.regions = {
            name: [a - x0, b - y0, c - x0, d - y0]
            for name, (a, b, c, d) in self.regions.items()
        }
        return self.image, {
            "renderer": VERSION,
            "style": self.style,
            "image_size": list(self.image.size),
            "no_truncation": True,
            "pixels_sha256": hashlib.sha256(self.image.tobytes()).hexdigest(),
            "entities": self.entities,
            "labels": self.labels,
            "state_details": self.details,
            "regions": self.regions,
            "visual_facts": [list(a) for a in sorted(self.represented)],
            "rule_facts": [list(a) for a in sorted(self.facts - self.represented)],
            "initial_facts_accounted_for": len(self.facts),
        }


def dynamic_predicates(domain):
    result = set()

    def visit(node):
        if not isinstance(node, list) or not node:
            return
        if node[0] in {"and", "not"}:
            for child in node[1:]:
                visit(child)
        elif node[0] == "when":
            visit(node[2])
        elif node[0] == "forall":
            visit(node[2])
        else:
            result.add(node[0])

    for action in parse_sexpr(domain):
        if isinstance(action, list) and action[:1] == [":action"]:
            visit(action[action.index(":effect") + 1])
    return result


def sprite(s, kind, center, size, accent):
    """Material, shadows and silhouettes; no random changes to scene semantics."""
    d = s.draw
    x, y = center
    z = size

    def rect(box, fill, outline=None, width=2, radius=4):
        d.rounded_rectangle(
            tuple(box),
            min(radius, max(0, min(box[2] - box[0], box[3] - box[1]) / 2 - 1)),
            fill=fill,
            outline=outline,
            width=width,
        )

    d.ellipse((x - z * 0.46, y + z * 0.28, x + z * 0.48, y + z * 0.46), fill="#7d8987")
    if kind in {"crate", "package", "block", "part", "board", "relic"}:
        fill = "#b8864b" if kind in {"crate", "package", "board"} else accent
        box = (x - z * 0.43, y - z * 0.34, x + z * 0.36, y + z * 0.30)
        rect(box, tint(fill, -18), "#614c35", 2, 3)
        d.polygon(
            [
                (box[0], box[1]),
                (x - z * 0.32, y - z * 0.46),
                (x + z * 0.47, y - z * 0.46),
                (box[2], box[1]),
            ],
            fill=tint(fill, 25),
        )
        d.polygon(
            [
                (box[2], box[1]),
                (x + z * 0.47, y - z * 0.46),
                (x + z * 0.47, y + z * 0.18),
                (box[2], box[3]),
            ],
            fill=tint(fill, -36),
        )
        if kind in {"crate", "package", "board"}:
            for h in (-0.19, 0.0, 0.19):
                d.line(
                    (box[0] + 3, y + h * z, box[2] - 3, y + h * z),
                    fill="#986b3e",
                    width=2,
                )
        if kind == "part":
            d.ellipse(
                (x - z * 0.16, y - z * 0.19, x + z * 0.12, y + z * 0.1),
                fill="#3e555e",
                outline="#bfcccb",
                width=3,
            )
            for dx, dy in ((-0.31, -0.22), (0.25, -0.22), (-0.31, 0.2), (0.25, 0.2)):
                d.ellipse(
                    (x + dx * z - 2, y + dy * z - 2, x + dx * z + 2, y + dy * z + 2),
                    fill="#d8dfd9",
                )
        if kind == "crate":
            d.line(
                (box[0] + 5, box[1] + 5, box[2] - 5, box[3] - 5),
                fill="#dab27b",
                width=max(2, int(z * 0.07)),
            )
            d.line(
                (box[0] + 5, box[3] - 5, box[2] - 5, box[1] + 5),
                fill="#dab27b",
                width=max(2, int(z * 0.07)),
            )
        if kind == "package":
            d.rectangle((x - z * 0.08, box[1], x + z * 0.06, box[3]), fill="#dfc392")
            rect(
                (x - z * 0.32, y - z * 0.12, x - z * 0.12, y + z * 0.05),
                "#f6efda",
                radius=1,
            )
        if kind == "relic":
            d.ellipse(
                (x - z * 0.15, y - z * 0.23, x + z * 0.15, y + z * 0.15),
                fill="#ead09b",
                outline="#a68349",
                width=3,
            )
    elif kind in {"player", "worker", "passenger"}:
        d.ellipse(
            (x - z * 0.17, y - z * 0.48, x + z * 0.17, y - z * 0.16),
            fill="#d9b98b",
            outline="#81654c",
            width=2,
        )
        rect(
            (x - z * 0.26, y - z * 0.15, x + z * 0.26, y + z * 0.23),
            accent,
            "#34556a",
            2,
            8,
        )
        d.line(
            (x - z * 0.12, y + z * 0.2, x - z * 0.15, y + z * 0.42),
            fill="#283a45",
            width=max(5, int(z * 0.12)),
        )
        d.line(
            (x + z * 0.12, y + z * 0.2, x + z * 0.15, y + z * 0.42),
            fill="#283a45",
            width=max(5, int(z * 0.12)),
        )
        if kind == "worker":
            d.arc(
                (x - z * 0.21, y - z * 0.55, x + z * 0.21, y - z * 0.10),
                180,
                360,
                fill="#e8b644",
                width=max(4, int(z * 0.09)),
            )
    elif kind in {"robot", "rover", "truck"}:
        for dx in (-0.34, 0.34):
            rect(
                (
                    x + dx * z - z * 0.08,
                    y - z * 0.1,
                    x + dx * z + z * 0.08,
                    y + z * 0.4,
                ),
                "#263b42",
                radius=6,
            )
        rect(
            (x - z * 0.31, y - z * 0.29, x + z * 0.31, y + z * 0.31),
            accent,
            "#314a55",
            3,
            7,
        )
        rect(
            (x - z * 0.23, y - z * 0.22, x + z * 0.23, y + z * 0.04),
            "#bfdce0",
            "#547c8d",
            2,
            4,
        )
        if kind == "rover":
            d.line((x, y - z * 0.2, x + z * 0.1, y - z * 0.55), fill="#435963", width=4)
            rect(
                (x - z * 0.04, y - z * 0.57, x + z * 0.25, y - z * 0.43),
                "#c1c9c7",
                radius=3,
            )
        if kind == "truck":
            rect(
                (x - z * 0.27, y + z * 0.02, x + z * 0.27, y + z * 0.38),
                "#d2bc8a",
                "#665e46",
                2,
                3,
            )
        if kind == "robot":
            for dx in (-0.18, 0.18):
                d.ellipse(
                    (
                        x + z * dx - 4,
                        y - z * 0.15 - 4,
                        x + z * dx + 4,
                        y - z * 0.15 + 4,
                    ),
                    fill="#1b5361",
                )
    elif kind == "ball":
        d.ellipse(
            (x - z * 0.34, y - z * 0.36, x + z * 0.34, y + z * 0.32),
            fill=tint(accent, -25),
            outline="#51616a",
            width=2,
        )
        d.ellipse((x - z * 0.27, y - z * 0.32, x + z * 0.27, y + z * 0.23), fill=accent)
        d.ellipse(
            (x - z * 0.19, y - z * 0.23, x - z * 0.02, y - z * 0.06),
            fill=tint(accent, 65),
        )
    elif kind in {"key", "spanner", "cutter", "pliers", "tool", "clamp"}:
        metal = "#bac8cc"
        if kind == "key":
            d.ellipse(
                (x - z * 0.4, y - z * 0.3, x - z * 0.04, y + z * 0.06),
                outline="#d4b65d",
                width=max(4, int(z * 0.08)),
            )
            d.line(
                (x - z * 0.09, y - z * 0.06, x + z * 0.35, y + z * 0.33),
                fill="#d4b65d",
                width=max(4, int(z * 0.09)),
            )
            d.line(
                (x + z * 0.2, y + z * 0.2, x + z * 0.28, y + z * 0.09),
                fill="#d4b65d",
                width=5,
            )
        elif kind == "clamp":
            d.arc(
                (x - z * 0.4, y - z * 0.4, x + z * 0.32, y + z * 0.31),
                70,
                290,
                fill=accent,
                width=max(8, int(z * 0.16)),
            )
            d.line(
                (x + z * 0.19, y - z * 0.38, x + z * 0.19, y + z * 0.37),
                fill=metal,
                width=6,
            )
            d.line(
                (x - z * 0.01, y + z * 0.34, x + z * 0.39, y + z * 0.34),
                fill="#354c56",
                width=5,
            )
        else:
            d.line(
                (x - z * 0.25, y + z * 0.34, x + z * 0.18, y - z * 0.21),
                fill=metal,
                width=max(6, int(z * 0.14)),
            )
            d.arc(
                (x - z * 0.02, y - z * 0.43, x + z * 0.4, y - z * 0.02),
                20,
                290,
                fill=metal,
                width=max(6, int(z * 0.12)),
            )
            if kind in {"pliers", "cutter"}:
                d.line(
                    (x + z * 0.25, y + z * 0.33, x - z * 0.03, y - z * 0.16),
                    fill=accent,
                    width=9,
                )
    elif kind == "nut":
        points = [
            (
                x + z * 0.33 * math.cos(i * math.pi / 3),
                y + z * 0.33 * math.sin(i * math.pi / 3),
            )
            for i in range(6)
        ]
        d.polygon(points, fill="#b7c4c5", outline="#53666a")
        d.ellipse(
            (x - z * 0.13, y - z * 0.13, x + z * 0.13, y + z * 0.13),
            fill="#526267",
            outline="#dae0de",
            width=3,
        )
    elif kind in {"printer", "machine"}:
        rect(
            (x - z * 0.43, y - z * 0.2, x + z * 0.43, y + z * 0.32),
            "#c5cdd0",
            "#566c76",
            3,
            6,
        )
        rect(
            (x - z * 0.34, y - z * 0.4, x + z * 0.33, y - z * 0.04),
            "#758b96",
            "#425e6b",
            2,
            5,
        )
        rect(
            (x - z * 0.26, y - z * 0.32, x + z * 0.25, y - z * 0.14),
            "#d8edf0",
            radius=2,
        )
        rect(
            (x - z * 0.3, y + z * 0.11, x + z * 0.3, y + z * 0.22), "#324550", radius=2
        )
        d.ellipse(
            (x + z * 0.25, y - z * 0.03, x + z * 0.32, y + z * 0.04), fill="#e4b451"
        )
    elif kind == "table":
        rect(
            (x - z * 0.43, y - z * 0.28, x + z * 0.43, y + z * 0.16),
            "#b9976c",
            "#685c45",
            3,
            4,
        )
        for dx in (-0.34, 0.34):
            d.line(
                (x + dx * z, y + z * 0.11, x + dx * z, y + z * 0.39),
                fill="#6b5944",
                width=7,
            )
    elif kind == "cartridge":
        rect(
            (x - z * 0.26, y - z * 0.4, x + z * 0.26, y + z * 0.3),
            "#46525a",
            "#26313a",
            2,
            4,
        )
        rect((x - z * 0.23, y - z * 0.21, x + z * 0.23, y + z * 0.15), accent, radius=1)
        d.rectangle(
            (x - z * 0.12, y + z * 0.27, x + z * 0.12, y + z * 0.4), fill="#d1b963"
        )
    elif kind in {"plate", "pan", "pot", "bowl"}:
        d.ellipse(
            (x - z * 0.45, y - z * 0.3, x + z * 0.45, y + z * 0.33),
            fill="#aeb6b5",
            outline="#6e8389",
            width=3,
        )
        d.ellipse(
            (x - z * 0.4, y - z * 0.29, x + z * 0.4, y + z * 0.2),
            fill="#f0ece2" if kind == "plate" else "#bec9cc",
            outline="#8c9b9f",
            width=3,
        )
        if kind in {"pan", "pot"}:
            d.line(
                (x + z * 0.35, y, x + z * 0.6, y + z * 0.16), fill="#394e54", width=8
            )
    elif kind == "sink":
        rect(
            (x - z * 0.45, y - z * 0.3, x + z * 0.45, y + z * 0.32),
            "#9daeb3",
            "#52696f",
            2,
            6,
        )
        rect(
            (x - z * 0.34, y - z * 0.23, x + z * 0.34, y + z * 0.24),
            "#7296a2",
            "#d0dce0",
            3,
            8,
        )
        d.arc(
            (x - z * 0.14, y - z * 0.52, x + z * 0.24, y - z * 0.12),
            170,
            360,
            fill="#d8e2df",
            width=7,
        )
    elif kind in {"oven", "rack", "chop"}:
        if kind == "chop":
            rect(
                (x - z * 0.45, y - z * 0.25, x + z * 0.45, y + z * 0.3),
                "#c9a474",
                "#826b47",
                3,
                5,
            )
            d.polygon(
                [
                    (x - z * 0.23, y - z * 0.12),
                    (x + z * 0.3, y + z * 0.02),
                    (x + z * 0.28, y + z * 0.1),
                    (x - z * 0.24, y - z * 0.04),
                ],
                fill="#d2dfe0",
            )
        else:
            rect(
                (x - z * 0.4, y - z * 0.4, x + z * 0.4, y + z * 0.35),
                "#91a3a8",
                "#4b6571",
                3,
                6,
            )
            rect(
                (x - z * 0.29, y - z * 0.2, x + z * 0.29, y + z * 0.22),
                "#304d5a",
                "#b1c5c9",
                3,
                4,
            )
            for h in (-0.08, 0.07, 0.18):
                d.line(
                    (x - z * 0.25, y + z * h, x + z * 0.25, y + z * h),
                    fill="#c2d0d0",
                    width=2,
                )
    elif kind.startswith("ingredient:"):
        name = kind.split(":", 1)[1]
        if any(n in name for n in ("tomato", "pepper", "carrot")):
            fill = "#d85e3e"
        elif any(n in name for n in ("herb", "lettuce", "bean")):
            fill = "#639857"
        elif any(n in name for n in ("onion", "potato", "dough", "flour", "pasta")):
            fill = "#dcc994"
        else:
            fill = "#d6af75"
        d.ellipse(
            (x - z * 0.35, y - z * 0.29, x + z * 0.35, y + z * 0.3),
            fill=tint(fill, -22),
            outline="#655e42",
            width=2,
        )
        d.ellipse((x - z * 0.3, y - z * 0.26, x + z * 0.27, y + z * 0.2), fill=fill)
        d.line((x, y - z * 0.27, x + z * 0.09, y - z * 0.44), fill="#476d3f", width=4)
    elif kind == "satellite":
        for dx in (-0.4, 0.4):
            rect(
                (
                    x + dx * z - z * 0.21,
                    y - z * 0.25,
                    x + dx * z + z * 0.21,
                    y + z * 0.24,
                ),
                "#284f78",
                "#779eae",
                2,
                2,
            )
            for k in (-0.08, 0.06):
                d.line(
                    (
                        x + dx * z - z * 0.18,
                        y + z * k,
                        x + dx * z + z * 0.18,
                        y + z * k,
                    ),
                    fill="#92b8c6",
                    width=2,
                )
        rect(
            (x - z * 0.17, y - z * 0.34, x + z * 0.17, y + z * 0.34),
            "#d4b785",
            "#8b7657",
            3,
            4,
        )
        d.arc(
            (x - z * 0.15, y - z * 0.55, x + z * 0.15, y - z * 0.18),
            10,
            170,
            fill="#d6e1dc",
            width=8,
        )
    elif kind in {"planet", "star", "camera", "lander"}:
        if kind in {"camera", "lander"}:
            rect(
                (x - z * 0.3, y - z * 0.28, x + z * 0.3, y + z * 0.22),
                "#c5ccc8",
                "#5e7378",
                3,
                5,
            )
            d.ellipse(
                (x - z * 0.16, y - z * 0.16, x + z * 0.16, y + z * 0.16),
                fill="#284d64",
                outline="#829aa4",
                width=4,
            )
        elif kind == "star":
            points = [
                (
                    x
                    + z
                    * (0.38 if i % 2 == 0 else 0.16)
                    * math.cos(i * math.pi / 5 - math.pi / 2),
                    y
                    + z
                    * (0.38 if i % 2 == 0 else 0.16)
                    * math.sin(i * math.pi / 5 - math.pi / 2),
                )
                for i in range(10)
            ]
            d.polygon(points, fill="#ecd39a")
        else:
            d.ellipse(
                (x - z * 0.36, y - z * 0.36, x + z * 0.36, y + z * 0.36),
                fill=accent,
                outline=tint(accent, 35),
                width=3,
            )
            d.arc(
                (x - z * 0.46, y - z * 0.12, x + z * 0.46, y + z * 0.17),
                5,
                175,
                fill="#d4c0a4",
                width=5,
            )
    elif kind in {"case", "carrier", "checkpoint"}:
        rect(
            (x - z * 0.38, y - z * 0.29, x + z * 0.38, y + z * 0.30),
            accent,
            "#415963",
            3,
            6,
        )
        if kind == "case":
            d.arc(
                (x - z * 0.16, y - z * 0.45, x + z * 0.16, y - z * 0.15),
                180,
                360,
                fill="#354b53",
                width=5,
            )
            d.line((x - z * 0.37, y, x + z * 0.37, y), fill="#d1b881", width=3)
        if kind == "checkpoint":
            d.line((x, y - z * 0.25, x, y + z * 0.27), fill="#e4e2c9", width=4)
            d.polygon(
                [(x, y - z * 0.27), (x + z * 0.25, y - z * 0.22), (x, y - z * 0.05)],
                fill="#eadba2",
            )
    else:
        raise ValueError("Unknown scene sprite: " + kind)


def block_scene(s, scenario):
    state = s.state
    game = s.game
    s.texture((20, 105, 1420, 995), "wood")
    s.draw.rectangle((35, 800, 1405, 845), fill="#685139")
    if game == "blocks":
        roots = [a[0] for a in state["on_table"]]
        piles = stacks(state, roots)
        piles = [[base, *p] for base, p in zip(roots, piles)]
        n = max(len(p) for p in piles)
        unit = min(108, 590 / n)
        width = min(145, 1130 / max(len(piles), 1))
        for i, pile in enumerate(piles):
            x = 130 + i * (1180 / max(len(piles), 1))
            for j, obj in enumerate(pile):
                bottom = 800 - j * unit
                top = bottom - unit
                half = min(width * 0.45, unit * 0.65)
                fill = color(obj)
                s.draw.rectangle(
                    (x - half, top, x + half, bottom),
                    fill=tint(fill, -18),
                    outline="#435156",
                    width=2,
                )
                s.draw.polygon(
                    [
                        (x + half, top),
                        (x + half + 10, top - 8),
                        (x + half + 10, bottom - 8),
                        (x + half, bottom),
                    ],
                    fill=tint(fill, -35),
                )
                s.draw.polygon(
                    [
                        (x - half, top),
                        (x - half + 10, top - 8),
                        (x + half + 10, top - 8),
                        (x + half, top),
                    ],
                    fill=tint(fill, 24),
                )
                s.entities.append(
                    {
                        "id": obj,
                        "kind": "block",
                        "bbox": [
                            round(x - half),
                            round(top),
                            round(x + half + 10),
                            round(bottom),
                        ],
                        "location": "table" if j == 0 else pile[j - 1],
                        "attributes": {"clear": s.present("clear", obj)},
                    }
                )
                s.label(
                    (x, top + unit * 0.35),
                    obj,
                    min(26, int(unit * 0.52)),
                    True,
                    "#fff9ea",
                    "center",
                )
                s.fact("on_table", obj) if j == 0 else s.fact("on", obj, pile[j - 1])
                if s.present("clear", obj):
                    s.fact("clear", obj)
        s.fact("arm_empty")
        s.draw.line(
            [(150, 955), (190, 910), (220, 930), (240, 910)], fill="#667f8d", width=22
        )
        s.draw.line((240, 910, 255, 928), fill="#263e4b", width=6)
    else:
        pegs = ["peg1", "peg2", "peg3"]
        piles = stacks(state, pegs)
        n = max(len(p) for p in piles)
        unit = min(65, 510 / max(n, 1))
        for i, (peg, pile) in enumerate(zip(pegs, piles)):
            x = 260 + i * 460
            s.draw.rounded_rectangle(
                (x - 7, 210, x + 7, 805), 5, fill="#897555", outline="#4e4435", width=3
            )
            s.draw.ellipse(
                (x - 142, 773, x + 142, 815), fill="#8e7049", outline="#594834", width=3
            )
            for j, obj in enumerate(pile):
                num = int(obj[1:])
                length = 95 + num * 20
                y = 773 - j * unit
                fill = color(obj)
                s.draw.rounded_rectangle(
                    (x - length / 2, y - unit, x + length / 2, y),
                    9,
                    fill=tint(fill, -20),
                    outline="#494e45",
                    width=2,
                )
                s.draw.ellipse(
                    (x - length / 2, y - unit - 7, x + length / 2, y - unit + 15),
                    fill=fill,
                    outline=tint(fill, 32),
                    width=2,
                )
                s.label((x, y - unit * 0.68), obj, 24, True, "#fff7e9", "center")
                s.entities.append(
                    {
                        "id": obj,
                        "kind": "disk",
                        "bbox": [
                            int(x - length / 2),
                            int(y - unit - 7),
                            int(x + length / 2),
                            int(y),
                        ],
                        "location": peg if j == 0 else pile[j - 1],
                        "attributes": {"size": num, "clear": s.present("clear", obj)},
                    }
                )
                s.fact("on", obj, peg if j == 0 else pile[j - 1])
                if s.present("clear", obj):
                    s.fact("clear", obj)
            s.label((x, 840), peg, 30, True, anchor="center")
            s.entities.append(
                {
                    "id": peg,
                    "kind": "peg",
                    "bbox": [x - 10, 210, x + 10, 805],
                    "location": None,
                    "attributes": {"clear": not pile},
                }
            )
            if s.present("clear", peg):
                s.fact("clear", peg)


def puzzle_scene(s, scenario):
    cells = [a[0] for a in s.state["position"]]
    h = max(int(c[1:3]) for c in cells)
    w = max(int(c[4:6]) for c in cells)
    unit = min(255, 790 / max(h, w))
    ox = (WIDTH - w * unit) / 2
    oy = 170
    s.texture((25, 100, 1415, 995), "darkwood")
    at = {c: t for t, c in s.state["at"]}
    for r in range(h):
        for c in range(w):
            name = f"r{r + 1:02d}c{c + 1:02d}"
            x = ox + c * unit
            y = oy + r * unit
            s.regions[name] = [round(x), round(y), round(x + unit), round(y + unit)]
            if name in at:
                s.texture((x + 6, y + 6, x + unit - 6, y + unit - 6), "wood")
                s.draw.line(
                    (x + 7, y + 7, x + unit - 7, y + 7), fill="#e1c296", width=5
                )
                t = at[name]
                s.label(
                    (x + unit / 2, y + unit * 0.32), t, 64, True, "#483824", "center"
                )
                s.entities.append(
                    {
                        "id": t,
                        "kind": "sliding-tile",
                        "bbox": [
                            int(x + 6),
                            int(y + 6),
                            int(x + unit - 6),
                            int(y + unit - 6),
                        ],
                        "location": name,
                        "attributes": {},
                    }
                )
                s.fact("at", t, name)
            else:
                s.draw.rectangle(
                    (x + 6, y + 6, x + unit - 6, y + unit - 6), fill="#332e25"
                )
                s.label(
                    (x + unit / 2, y + unit * 0.4),
                    "BLANK",
                    29,
                    True,
                    "#bcaa88",
                    "center",
                )
                s.fact("empty", name)
            s.label(
                (x + unit / 2, y + unit - 37),
                name,
                24,
                False,
                "#f6eddc" if name not in at else "#4f402a",
                "center",
            )


def cards_scene(s, scenario):
    state = s.state
    s.texture((20, 100, 1420, 1000), "felt")
    ranks = dict(state["value"])
    suits = dict(state["hassuit"])
    bottoms = [a[0] for a in state["bottomcol"]]
    empty = int(state["colspace"][0][0].replace("coln", ""))
    columns = len(bottoms) + empty
    space = int(state["cellspace"][0][0].replace("celln", ""))
    occupied = [a[0] for a in state["incell"]]
    homes = [a[0] for a in state["home"]]

    def card(name, x, y):
        red = suits[name] in {"h", "d"}
        fill = "#ad4939" if red else "#293d42"
        s.draw.rounded_rectangle(
            (x, y, x + 146, y + 183), 9, fill="#fbf7e8", outline="#b6ad90", width=3
        )
        rank = ranks[name].removeprefix("n")
        s.label((x + 13, y + 10), name, 25, True, fill)
        s.label(
            (x + 74, y + 73), rank + " " + suits[name].upper(), 38, True, fill, "center"
        )
        s.entities.append(
            {
                "id": name,
                "kind": "playing-card",
                "bbox": [int(x), int(y), int(x + 146), int(y + 183)],
                "location": None,
                "attributes": {"rank": ranks[name], "suit": suits[name]},
            }
        )

    for i in range(space + len(occupied)):
        x = 45 + i * 175
        s.draw.rounded_rectangle((x, 147, x + 150, 337), 9, outline="#b5c4ae", width=3)
        s.label((x + 75, 115), "FREECELL", 20, True, "#e1e5cf", "center")
        if i < len(occupied):
            card(occupied[i], x + 2, 150)
            s.fact("incell", occupied[i])
    for i, home in enumerate(homes):
        x = 1020 + i * 180
        if ranks[home] == "n0":
            s.draw.rounded_rectangle(
                (x, 150, x + 146, 333), 9, outline="#adc0ac", width=3
            )
            s.label((x + 73, 208), "EMPTY", 24, True, "#e2ead4", "center")
            s.label((x + 73, 255), home, 24, True, "#e2ead4", "center")
            s.entities.append(
                {
                    "id": home,
                    "kind": "foundation-base",
                    "bbox": [x, 150, x + 146, 333],
                    "location": None,
                    "attributes": {"rank": "n0", "suit": suits[home]},
                }
            )
        else:
            card(home, x, 150)
        s.label((x + 73, 115), "FOUNDATION", 20, True, "#e1e5cf", "center")
        s.fact("home", home)
    piles = [[base, *p] for base, p in zip(bottoms, stacks(state, bottoms))]
    for i in range(columns):
        x = 55 + i * (1295 / max(columns, 1))
        y = 398
        s.draw.rounded_rectangle((x, y, x + 150, 970), 9, outline="#8fb4a0", width=2)
        if i >= len(piles):
            continue
        pile = piles[i]
        step = min(65, 390 / max(len(pile) - 1, 1))
        for j, name in enumerate(pile):
            card(name, x + 2, y + j * step)
            s.entities[-1]["location"] = "tableau" if j == 0 else pile[j - 1]
            s.fact("bottomcol", name) if j == 0 else s.fact("on", name, pile[j - 1])
            if s.present("clear", name):
                s.fact("clear", name)
    for pred in ("cellspace", "colspace"):
        for args in state[pred]:
            s.fact(pred, *args)


def grid_scene(s, scenario):
    grid = scenario["scene"]
    state = s.state
    game = s.game
    h, w = grid["height"], grid["width"]
    unit = min((WIDTH - 140) / w, (HEIGHT - 100) / h)
    ox = (WIDTH - w * unit) / 2
    oy = (HEIGHT - h * unit) / 2
    cells = set(grid["cells"])
    holes = set(grid.get("holes", []))
    goals = set(grid.get("goals", []))
    positions = {}
    for r in range(h):
        s.label(
            (ox - 14, oy + r * unit + unit * 0.45),
            f"r{r + 1:02d}",
            22,
            True,
            anchor="right",
        )
        for c in range(w):
            cell = f"r{r + 1:02d}c{c + 1:02d}"
            x, y = ox + c * unit, oy + r * unit
            if r == 0:
                s.label(
                    (x + unit / 2, oy - 31), f"c{c + 1:02d}", 22, True, anchor="center"
                )
            box = (x + 1, y + 1, x + unit - 1, y + unit - 1)
            if cell not in cells:
                s.texture(box, "brick")
                continue
            positions[cell] = (x + unit / 2, y + unit / 2)
            s.regions[cell] = list(map(round, (x, y, x + unit, y + unit)))
            s.texture(
                box,
                "water"
                if cell in holes
                else "ice"
                if game == "frozenlake"
                else "wood"
                if game == "sokoban"
                else "tile",
            )
            if s.present("safe", cell):
                s.fact("safe", cell)
            if s.present("fragile", cell):
                s.fact("fragile", cell)
                s.draw.line(
                    [
                        (x + unit * 0.2, y + unit * 0.05),
                        (x + unit * 0.5, y + unit * 0.3),
                        (x + unit * 0.35, y + unit * 0.75),
                    ],
                    fill="#377998",
                    width=4,
                )
                s.draw.line(
                    (x + unit * 0.5, y + unit * 0.3, x + unit * 0.8, y + unit * 0.1),
                    fill="#377998",
                    width=3,
                )
            if cell in goals:
                s.draw.ellipse(
                    (
                        x + unit * 0.18,
                        y + unit * 0.18,
                        x + unit * 0.82,
                        y + unit * 0.82,
                    ),
                    outline="#d49b13",
                    width=6,
                )
            if s.present("delivery", cell):
                s.draw.rectangle(
                    (x + 5, y + 5, x + unit - 5, y + unit - 5),
                    outline="#318658",
                    width=6,
                )
                s.fact("delivery", cell)
    passages = {frozenset((a, b)) for a, b, d in grid["edges"]}
    for cell, (x, y) in positions.items():
        r, c = map(int, re.findall(r"\d+", cell))
        for dr, dc in ((0, 1), (1, 0)):
            other = f"r{r + dr:02d}c{c + dc:02d}"
            if other in positions and frozenset((cell, other)) not in passages:
                line = (
                    (x + unit / 2, y - unit / 2, x + unit / 2, y + unit / 2)
                    if dc
                    else (x - unit / 2, y + unit / 2, x + unit / 2, y + unit / 2)
                )
                s.draw.line(line, fill=s.palette["wall"], width=12)
    occupancy = defaultdict(list)
    for pred in ("at", "tool_at", "key_at", "checkpoint_at"):
        for obj, cell in state[pred]:
            occupancy[cell].append((obj, pred))
            s.fact(pred, obj, cell)
    heading = state["facing"][0][0] if state["facing"] else None
    parent = {a: b for a, b in state["depends_on"]}
    for cell, items in sorted(occupancy.items()):
        cx, cy = positions[cell]
        x0, y0 = cx - unit / 2, cy - unit / 2
        packages = [name for name, p in items if s.present("package", name)]
        if packages:

            def package_box(name, box):
                x, y, bx, by = box
                children = [n for n in packages if parent.get(n) == name]
                s.draw.rectangle(
                    box, fill=tint(s.palette["floor"], -8), outline="#8e6b42", width=3
                )
                tag = "p" + name.removeprefix("package")
                s.label((x + 5, y + 3), tag, 17, True)
                if s.present("taped", name):
                    s.draw.line(
                        (bx - 20, y + 4, bx - 20, y + 19), fill="#bba051", width=7
                    )
                    s.fact("taped", name)
                if s.present("tape_cut", name):
                    s.fact("tape_cut", name)
                if s.present("clipped", name):
                    s.draw.arc(
                        (bx - 13, y + 3, bx - 3, y + 16),
                        0,
                        320,
                        fill="#657881",
                        width=4,
                    )
                    s.fact("clipped", name)
                if s.present("open", name):
                    s.draw.line((x, y, x - 3, y - 5), fill="#8e6b42", width=3)
                    s.fact("open", name)
                s.entities.append(
                    {
                        "id": name,
                        "kind": "package",
                        "bbox": list(map(round, box)),
                        "location": cell,
                        "attributes": {
                            k: s.present(k, name)
                            for k in ("taped", "tape_cut", "clipped", "open")
                        },
                    }
                )
                if name in parent:
                    s.fact("depends_on", name, parent[name])
                if children:
                    cw = (bx - x - 12) / len(children)
                    for j, child in enumerate(children):
                        package_box(
                            child,
                            (x + 6 + j * cw, y + 24, x + 6 + (j + 1) * cw - 3, by - 5),
                        )

            roots = [n for n in packages if n not in parent]
            for j, name in enumerate(roots):
                package_box(
                    name,
                    (
                        x0 + 6 + j * (unit - 12) / len(roots),
                        y0 + 10,
                        x0 + 6 + (j + 1) * (unit - 12) / len(roots) - 4,
                        y0 + unit - 10,
                    ),
                )
            items = [a for a in items if a[0] not in packages]
        cols = 1 if len(items) == 1 else 2
        rows = math.ceil(len(items) / cols)
        for j, (name, pred) in enumerate(items):
            px = x0 + (j % cols + 0.5) * unit / cols
            py = y0 + (j // cols + 0.47) * unit / max(rows, 1)
            if rows > 1:
                py = y0 + (j // cols + 0.5) * unit / rows - 12
            kind = (
                "player"
                if name == "player"
                else "crate"
                if game == "sokoban"
                else "table"
                if name.startswith("table")
                else "printer"
                if name.startswith("printer")
                else "cartridge"
                if name.startswith("cartridge")
                else "key"
                if pred == "key_at"
                else "checkpoint"
                if pred == "checkpoint_at"
                else "plate"
                if name.startswith("plate")
                else "ingredient:" + name
                if s.present("ingredient", name)
                else "sink"
                if "wash" in name
                else "chop"
                if any(k in name for k in ("chop", "peel"))
                else "oven"
                if any(k in name for k in ("bake", "roast"))
                else "rack"
                if "cool" in name
                else "pot"
                if any(k in name for k in ("boil", "soup"))
                else "pan"
                if "fry" in name
                else "bowl"
                if "mix" in name
                else name
                if name in {"cutter", "pliers"}
                else "tool"
            )
            size = min(unit / cols * 0.64, unit / max(rows, 1) * 0.56, 104)
            if rows > 1:
                # Reserve the label height and clearance from the cell boundary.
                size = min(size, (unit / rows - 30) / 1.2)
            attributes = {
                k: s.present(k, name)
                for k in ("clean", "powered", "calibrated", "mounted", "table_free")
                if k in dynamic_predicates(s.domain)
            }
            s.object(name, kind, (px, py), size, cell, attributes, label=False)
            if name != "player":
                short = (
                    name.replace("checkpoint", "cp")
                    .replace("_station", "")
                    .replace("cartridge", "c")
                    .replace("printer", "p")
                    .replace("table", "t")
                )
                s.label(
                    (px, py + size * 0.61 + 2),
                    short,
                    18 if len(short) < 10 else 14,
                    True,
                    anchor="center",
                )
            for k, value in attributes.items():
                if value:
                    s.fact(k, name)
            if s.present("clean", name):
                s.draw.ellipse(
                    (
                        px + size * 0.34,
                        py - size * 0.45,
                        px + size * 0.5,
                        py - size * 0.29,
                    ),
                    fill="#36a563",
                )
            if s.present("ingredient", name):
                stage = next(v for n, v in state["stage"] if n == name)
                s.fact("stage", name, stage)
                s.label(
                    (px - size * 0.48, py - size * 0.5),
                    stage.removeprefix("phase"),
                    15,
                    True,
                )
            if s.present("printer", name):
                s.draw.ellipse(
                    (px + size * 0.23, py - 4, px + size * 0.34, py + 4),
                    fill="#42a66b" if s.present("powered", name) else "#344550",
                )
                count = next(v for n, v in state["paper_level"] if n == name)
                s.fact("paper_level", name, count)
                s.label(
                    (px + size * 0.3, py - size * 0.5),
                    count.removeprefix("n"),
                    16,
                    True,
                )
            if s.present("on_goal", name):
                s.fact("on_goal", name)
            if name == "player" and heading:
                dx, dy = {
                    "north": (0, -1),
                    "south": (0, 1),
                    "east": (1, 0),
                    "west": (-1, 0),
                }[heading]
                s.arrow(
                    (px, py), (px + dx * unit * 0.28, py + dy * unit * 0.28), width=6
                )
                s.fact("facing", heading)
    grouped_doors = defaultdict(list)
    seen_doors = set()
    for a, b, door in state["edge_door"]:
        if door not in seen_doors:
            grouped_doors[tuple(sorted((a, b)))].append(door)
            seen_doors.add(door)
    for (a, b), doors in grouped_doors.items():
        ax, ay = positions[a]
        bx, by = positions[b]
        for j, door in enumerate(doors):
            delta = (j - (len(doors) - 1) / 2) * 48
            mx, my = (ax + bx) / 2, (ay + by) / 2
            if ax == bx:
                mx += delta
            else:
                my += delta
            label_width = s.draw.textlength(door, font=font(15, True))
            half_width = max(15, label_width / 2 + 5)
            box = [round(mx - half_width), round(my - 15),
                   round(mx + half_width), round(my + 15)]
            s.draw.rectangle(box, fill="#dfc392", outline="#695b43", width=3)
            s.label((mx, my - 6), door, 15, True, anchor="center")
            s.entities.append(
                {
                    "id": door,
                    "kind": "door",
                    "bbox": box,
                    "location": [a, b],
                    "attributes": {"unlocked": s.present("unlocked", door)},
                }
            )
            if s.present("unlocked", door):
                s.fact("unlocked", door)
    for (cell,) in state["empty"]:
        s.fact("empty", cell)
    if state["arm_empty"]:
        player = next(e for e in s.entities if e["id"] == "player")
        x = (player["bbox"][0] + player["bbox"][2]) / 2
        y = (player["bbox"][1] + player["bbox"][3]) / 2
        s.draw.arc((x - 17, y + 10, x + 17, y + 27), 0, 180, fill="#4c676d", width=3)
        s.fact("arm_empty")


def room_scene(s, scenario):
    state = s.state
    game = s.game
    rooms = [a[0] for a in state["room" if game == "gripper" else "location"]]
    cols = min(3, len(rooms))
    rows = math.ceil(len(rooms) / cols)
    pw = 1300 / cols
    ph = 820 / rows
    for i, room in enumerate(rooms):
        x = 60 + (i % cols) * pw
        y = 90 + (i // cols) * ph
        s.texture((x, y, x + pw - 18, y + ph - 18), "wood")
        s.draw.rectangle(
            (x, y, x + pw - 18, y + ph - 18), outline=s.palette["wall"], width=5
        )
        s.label((x + 15, y + 13), room, 27, True)
        s.regions[room] = list(map(round, (x, y, x + pw - 18, y + ph - 18)))
        objects = [n for n, r in state["at"] if r == room and not s.present("in", n)]
        has_case = game == "briefcase" and s.present("is_at", room)
        object_height = ph - (240 if has_case else 160)
        for j, name in enumerate(objects):
            ncols = min(4, len(objects))
            nrows = math.ceil(len(objects) / ncols)
            spacing = (pw - 50) / ncols
            px = x + 25 + (j % ncols + 0.5) * spacing
            py = y + 80 + (j // ncols + 0.5) * object_height / nrows
            s.object(
                name,
                "ball" if game == "gripper" else "relic",
                (px, py),
                min(72, spacing * 0.65, object_height / nrows * 0.5),
                room,
                {},
            )
            s.fact("at", name, room)
        if game == "gripper" and s.present("at_robby", room):
            rx, ry = x + pw * 0.7, y + ph - 95
            s.object(
                "robby",
                "robot",
                (rx, ry),
                70,
                room,
                {
                    "left_free": s.present("free", "left"),
                    "right_free": s.present("free", "right"),
                },
                label=False,
            )
            s.fact("at_robby", room)
            for j, grip in enumerate(("left", "right")):
                s.draw.arc(
                    (rx - 70 + j * 105, ry - 12, rx - 35 + j * 105, ry + 25),
                    0,
                    270,
                    fill="#344f5e",
                    width=5,
                )
                if s.present("free", grip):
                    s.fact("free", grip)
        if has_case:
            # A cutaway keeps every contained item's ID readable inside the case.
            contents = state["in"]
            rx, ry = x + (pw - 18) / 2, y + ph - 95
            case_width = max(120, len(contents) * 90 + 24)
            box = [round(rx - case_width / 2), round(ry - 50),
                   round(rx + case_width / 2), round(ry + 48)]
            s.draw.rounded_rectangle(tuple(box), radius=8, fill=s.palette["ground"],
                                     outline="#315d72", width=4)
            s.draw.arc((rx - 20, ry - 68, rx + 20, ry - 32), 180, 360,
                       fill="#315d72", width=4)
            s.label((rx, ry + 53), "briefcase", 22, True, anchor="center")
            s.entities.append({"id": "briefcase", "kind": "case", "bbox": box,
                               "location": room, "attributes": {}})
            s.fact("is_at", room)
            for j, (name,) in enumerate(contents):
                s.object(
                    name,
                    "relic",
                    (rx + (j - (len(contents) - 1) / 2) * 90, ry - 12),
                    40,
                    "briefcase",
                    {"case_room": room},
                    label=True,
                )
                s.fact("in", name)
                s.fact("at", name, room)


def lift_scene(s, scenario):
    state = s.state
    floors = sorted(
        [a[0] for a in state["floor"]], key=lambda x: int(x[1:]), reverse=True
    )
    unit = 900 / len(floors)
    for i, floor in enumerate(floors):
        y = 70 + i * unit
        s.texture((70, y, 1000, y + unit - 4), "wood")
        s.draw.line(
            (70, y + unit - 4, 1350, y + unit - 4), fill=s.palette["wall"], width=5
        )
        s.label((82, y + unit * 0.4), floor, 25, True)
        s.regions[floor] = [70, round(y), 1350, round(y + unit - 4)]
        waiting = [p for p, f in state["origin"] if f == floor]
        for j, p in enumerate(waiting):
            x = 235 + j * 110
            s.object(
                p,
                "passenger",
                (x, y + unit * 0.36),
                min(unit * 0.43, 58),
                floor,
                {"boarded": False, "served": False},
                label=False,
            )
            target = next(f for n, f in state["destin"] if n == p)
            s.label((x, y + unit * 0.76), p + ">" + target, 19, True, anchor="center")
            s.fact("not_boarded", p)
            s.fact("not_served", p)
        if s.present("lift_at", floor):
            box = [1020, round(y + 4), 1340, round(y + unit - 8)]
            s.draw.rectangle(box, fill="#b6c8cf", outline="#526f7e", width=4)
            s.fact("lift_at", floor)
            s.entities.append(
                {
                    "id": "lift",
                    "kind": "lift",
                    "bbox": box,
                    "location": floor,
                    "attributes": {},
                }
            )
    if state["free_slots"]:
        count = state["free_slots"][0][0]
        s.note((1110, 1010), count, [("free_slots", count)], 23)


def lights_scene(s, scenario):
    state = s.state
    names = [a[0] for a in state["light"]]
    h = max(int(n.split("_")[0][5:]) for n in names) + 1
    w = max(int(n.split("_")[1]) for n in names) + 1
    unit = min(920 / h, 1170 / w)
    ox = (WIDTH - w * unit) / 2 + 35
    oy = (HEIGHT - h * unit) / 2
    for r in range(h):
        panel = f"panel{r}"
        y = oy + r * unit
        connected = s.present("connected", panel)
        s.draw.line(
            (ox - 40, y + unit * 0.15, ox - 40, y + unit * 0.85),
            fill="#6080a1",
            width=6,
        )
        s.label((ox - 52, y + unit * 0.44), panel, 21, True, anchor="right")
        box = [
            round(ox - 48),
            round(y + unit * 0.15),
            round(ox - 32),
            round(y + unit * 0.85),
        ]
        s.entities.append(
            {
                "id": panel,
                "kind": "panel-connector",
                "bbox": box,
                "location": r,
                "attributes": {"connected": connected},
            }
        )
        if connected:
            s.fact("connected", panel)
        for c in range(w):
            light = f"light{r}_{c}"
            button = f"button{r}_{c}"
            x = ox + (c + 0.5) * unit
            cy = y + unit * 0.4
            rad = unit * 0.26
            lit = s.present("lit", light)
            s.draw.ellipse(
                (x - rad, cy - rad, x + rad, cy + rad),
                fill="#ffe57c" if lit else "#354b59",
                outline="#9ab1bc",
                width=5,
            )
            if lit:
                s.draw.ellipse(
                    (x - rad * 0.7, cy - rad * 0.8, x - rad * 0.15, cy - rad * 0.2),
                    fill="#fff6ce",
                )
                s.fact("lit", light)
            s.label((x, cy - rad - 24), light, 20, True, anchor="center")
            by = y + unit * 0.82
            s.draw.ellipse(
                (x - 14, by - 14, x + 14, by + 14),
                fill="#d18055",
                outline="#83613e",
                width=3,
            )
            s.label((x + 25, by - 10), button, 17)
            s.entities.extend(
                [
                    {
                        "id": light,
                        "kind": "lamp",
                        "bbox": list(
                            map(round, (x - rad, cy - rad, x + rad, cy + rad))
                        ),
                        "location": [r, c],
                        "attributes": {"lit": lit},
                    },
                    {
                        "id": button,
                        "kind": "button",
                        "bbox": list(map(round, (x - 14, by - 14, x + 14, by + 14))),
                        "location": [r, c],
                        "attributes": {"panel": panel},
                    },
                ]
            )
    if state["controller_free"]:
        s.draw.line((45, 55, 120, 55), fill="#6080a1", width=5)
        s.draw.rectangle((120, 46, 139, 64), fill="#6080a1")
        s.fact("controller_free")


def construction_scene(s, scenario):
    state = s.state
    game = s.game
    cells = [a[0] for a in state["position" if game == "termes" else "tile"]]
    if game == "termes":
        coords = {n: (int(n[1:3]) - 1, int(n[4:6]) - 1) for n in cells}
    else:
        raw = {n: tuple(map(int, n[5:].split("_"))) for n in cells}
        mr = max(r for r, c in raw.values())
        coords = {n: (mr - r, c - 1) for n, (r, c) in raw.items()}
    h = max(r for r, c in coords.values()) + 1
    w = max(c for r, c in coords.values()) + 1
    unit = min(880 / h, 1270 / w)
    ox = (WIDTH - w * unit) / 2
    oy = 50
    for name, (r, c) in coords.items():
        x, y = ox + c * unit, oy + r * unit
        s.texture((x + 2, y + 2, x + unit - 2, y + unit - 2), "tile")
        s.regions[name] = list(map(round, (x, y, x + unit, y + unit)))
        s.label((x + unit / 2, y + unit - 28), name, 19, anchor="center")
        if game == "termes":
            height = next(v for cell, v in state["height"] if cell == name)
            n = int(height[1:])
            s.fact("height", name, height)
            for k in range(n):
                sprite(
                    s,
                    "block",
                    (x + unit * 0.5, y + unit * 0.62 - k * 17),
                    unit * 0.45,
                    "#bd8e57",
                )
            s.label((x + 12, y + 10), height, 19, True)
            if s.present("is_depot", name):
                s.draw.rectangle(
                    (x + 5, y + 5, x + unit - 5, y + unit - 5),
                    outline="#268e57",
                    width=6,
                )
            if s.present("at", name):
                s.object(
                    "robot",
                    "robot",
                    (x + unit * 0.5, y + unit * 0.49),
                    unit * 0.48,
                    name,
                    {"carrying": bool(state["has_block"])},
                    label=False,
                )
                s.fact("at", name)
                if state["has_block"]:
                    s.object(
                        "carried-block",
                        "block",
                        (x + unit * 0.62, y + unit * 0.55),
                        unit * 0.18,
                        "robot",
                        {},
                        label=False,
                    )
                    s.fact("has_block")
        else:
            if s.present("clear", name):
                s.fact("clear", name)
            for cell, paint in state["painted"]:
                if cell == name:
                    s.draw.rectangle(
                        (x + 5, y + 5, x + unit - 5, y + unit - 34), fill=paint
                    )
                    s.fact("painted", cell, paint)
            for robot, cell in state["robot_at"]:
                if cell == name:
                    paint = next(v for rob, v in state["robot_has"] if rob == robot)
                    s.object(
                        robot,
                        "robot",
                        (x + unit * 0.5, y + unit * 0.47),
                        unit * 0.48,
                        name,
                        {"paint": paint},
                        label=False,
                    )
                    s.label((x + unit / 2, y + 8), robot, 20, True, anchor="center")
                    s.draw.ellipse(
                        (
                            x + unit * 0.73,
                            y + unit * 0.35,
                            x + unit * 0.87,
                            y + unit * 0.49,
                        ),
                        fill=paint,
                        outline="#8b9296",
                        width=2,
                    )
                    s.fact("robot_at", robot, cell)
                    s.fact("robot_has", robot, paint)
    if game == "floortile":
        for j, (paint,) in enumerate(state["available_color"]):
            x = 80 + j * 170
            s.draw.rectangle(
                (x, 980, x + 35, 1015), fill=paint, outline="#8b9296", width=2
            )
            s.note((x + 46, 985), paint, [("available_color", paint)], 21)
        for (robot,) in state["free_color"]:
            s.fact("free_color", robot)


def delivery_scene(s, scenario):
    state = s.state
    rooms = [a[0] for a in state["room"]]
    cols = 3
    rows = math.ceil(len(rooms) / cols)
    pw = 390
    ph = 850 / rows
    positions = {}
    for i, room in enumerate(rooms):
        row, col = divmod(i, cols)
        col = col if row % 2 == 0 else cols - 1 - col
        positions[room] = (120 + col * pw, 90 + row * ph)
    for i in range(len(rooms) - 1):
        a, b = rooms[i : i + 2]
        ax, ay = positions[a]
        bx, by = positions[b]
        s.draw.line(
            (ax + pw / 2, ay + ph / 2, bx + pw / 2, by + ph / 2),
            fill=s.palette["wall"],
            width=18,
        )
    for room, (x, y) in positions.items():
        box = [round(x), round(y), round(x + pw - 35), round(y + ph - 35)]
        s.texture(box, "wood")
        s.draw.rectangle(box, outline=s.palette["wall"], width=4)
        s.label((x + 14, y + 10), room, 25, True)
        s.regions[room] = box
        things = [(n, "key", "key_at") for n, r in state["key_at"] if r == room] + [
            (n, "relic", "relic_at") for n, r in state["relic_at"] if r == room
        ]
        for j, (name, kind, pred) in enumerate(things):
            px = x + 45 + (j % 4) * 76
            py = y + 80 + (j // 4) * 90
            s.object(name, kind, (px, py), 48, room, {}, label=True)
            s.fact(pred, name, room)
        if s.present("at_player", room):
            s.object(
                "player",
                "player",
                (x + pw - 95, y + ph - 100),
                70,
                room,
                {},
                label=False,
            )
            s.fact("at_player", room)
    for i in range(len(rooms) - 1):
        a, b = rooms[i : i + 2]
        ax, ay = positions[a]
        bx, by = positions[b]
        mx = (ax + bx) / 2 + pw / 2
        my = (ay + by) / 2 + ph / 2
        door = f"door{i}"
        box = [round(mx - 13), round(my - 23), round(mx + 13), round(my + 23)]
        s.draw.rectangle(box, fill="#b18e4e", outline="#5d553e", width=3)
        s.label((mx, my + 28), door, 17, True, anchor="center")
        s.entities.append(
            {
                "id": door,
                "kind": "door",
                "bbox": box,
                "location": [a, b],
                "attributes": {"unlocked": s.present("unlocked", door)},
            }
        )
        if s.present("unlocked", door):
            s.fact("unlocked", door)
    if state["hand_empty"]:
        player = next(e for e in s.entities if e["id"] == "player")
        x, y = player["bbox"][:2]
        s.draw.arc((x - 14, y + 40, x + 14, y + 58), 0, 180, fill="#475965", width=4)
        s.fact("hand_empty")


RENDERERS = {
    "blocks": block_scene,
    "hanoi": block_scene,
    "sliding_puzzle": puzzle_scene,
    "gripper": room_scene,
    "briefcase": room_scene,
    "miconic": lift_scene,
    "lights_out": lights_scene,
    "floortile": construction_scene,
    "termes": construction_scene,
    "key_delivery": delivery_scene,
    "freecell": cards_scene,
}


def visual_rejection(game, initial):
    if game == "blocks" and any(a[0] == "can_support" for a in initial):
        return "hidden_support_permissions"
    if game == "package":
        parent = {a[1]: a[2] for a in initial if a[0] == "depends_on"}
        for name in parent:
            depth = 0
            while name in parent:
                name = parent[name]
                depth += 1
                if depth > 2:
                    return "deep_nested_cutaway"
    return None


def render(scenario, style=None):
    s = Scene(scenario, style=style)
    if "scene" in scenario:
        grid_scene(s, scenario)
    else:
        RENDERERS[s.game](s, scenario)
    return s.finish()
