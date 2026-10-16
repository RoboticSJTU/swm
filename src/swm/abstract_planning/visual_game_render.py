"""Compact, auditable renders: readable labels, visible state, no prose panels."""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

from .render import font, tint
from .visual_games import COLORS, LETTERS, cell, rc, footprint, projection

THEMES = {
    "flat": dict(bg="#faf8f2", floor="#e8e2d6", ink="#172d3b", wall="#53626e", line="#8c999d"),
    "bevel": dict(bg="#f3f7fa", floor="#d9e6ec", ink="#142d3e", wall="#45647b", line="#839ca9"),
    "outline": dict(bg="#ffffff", floor="#f3f4f0", ink="#192a34", wall="#5b666b", line="#62747e"),
    "night": dict(bg="#202f42", floor="#bacbdb", ink="#10263a", wall="#415367", line="#63798d"),
}


class Canvas:
    def __init__(self, width, height, appearance):
        self.appearance = appearance; self.theme = THEMES[appearance["style"]]
        self.image = Image.new("RGB", (int(width), int(height)), self.theme["bg"])
        self.d = ImageDraw.Draw(self.image); self.labels = []; self.entities = []; self.probes = []
        self.regions = {}; self.layout = {}

    def label(self, xy, text, size=32, badge=True):
        x, y = map(round, xy); text = str(text); face = font(size, True)
        bbox = list(self.d.textbbox((x, y), text, font=face, anchor="mm"))
        ink = self.theme["ink"]; background = "#f8fafc" if badge else self.theme["bg"]
        if badge:
            a, b, c, d = bbox; self.d.rounded_rectangle((a-4, b-3, c+4, d+3), 4, fill=background)
        self.d.text((x, y), text, font=face, fill=ink, anchor="mm")
        self.labels.append(dict(text=text, bbox=bbox, font_size=size, xy=[x, y], ink=ink, background=background))
        return len(self.labels)-1

    def entity(self, ident, kind, bbox, label, **attributes):
        self.entities.append(dict(id=ident, kind=kind, bbox=list(bbox), label=label, **attributes))

    def probe(self, xy, color):
        self.probes.append(dict(xy=[int(xy[0]), int(xy[1])], color=color))

    def finish(self):
        # Crop only exterior background, preserving all grid cells and empty slots.
        pixels = np.asarray(self.image); background = np.array(self.image.getpixel((0, 0)))
        yy, xx = np.nonzero(np.any(pixels != background, axis=2))
        bounds = (max(0, int(xx.min())-18), max(0, int(yy.min())-18),
                  min(self.image.width, int(xx.max())+19), min(self.image.height, int(yy.max())+19))
        dx, dy = bounds[:2]; self.image = self.image.crop(bounds)
        for item in self.labels:
            item["xy"] = [item["xy"][0]-dx, item["xy"][1]-dy]
        for item in self.labels + self.entities:
            a, b, c, d = item["bbox"]; item["bbox"] = [a-dx, b-dy, c-dx, d-dy]
        for probe in self.probes: probe["xy"] = [probe["xy"][0]-dx, probe["xy"][1]-dy]
        for key, box in self.regions.items():
            a, b, c, d = box; self.regions[key] = [a-dx, b-dy, c-dx, d-dy]
        if "grid" in self.layout:
            self.layout["grid"]["left"] -= dx; self.layout["grid"]["top"] -= dy
        return self.image, dict(renderer="diverse-visual-games-v1", style=self.appearance["style"],
                                image_size=list(self.image.size), labels=self.labels, entities=self.entities,
                                probes=self.probes, regions=self.regions, layout=self.layout,
                                crop=list(bounds), no_truncation=True)


def board(spec, appearance, unit=None, extra=0):
    u = unit or appearance["scale"]; left = top = 62
    canvas = Canvas(left+spec["w"]*u+24+extra, top+spec["h"]*u+24, appearance)
    d, theme = canvas.d, canvas.theme
    for c in range(spec["w"]): canvas.label((left+(c+.5)*u, 28), f"c{c+1:02d}", 30)
    for r in range(spec["h"]): canvas.label((28, top+(r+.5)*u), f"r{r+1:02d}", 30)
    for r in range(spec["h"]):
        for c in range(spec["w"]):
            x, y = left+c*u, top+r*u
            d.rectangle((x, y, x+u, y+u), fill=theme["floor"], outline=theme["line"], width=2)
            if appearance["style"] == "bevel":
                d.line((x+3, y+3, x+u-3, y+3), fill="#f7fbfd", width=3)
            canvas.regions[cell(r, c)] = [x, y, x+u, y+u]
    canvas.layout["grid"] = dict(left=left, top=top, unit=u, h=spec["h"], w=spec["w"])
    return canvas, left, top, u


def numeric(task, state):
    spec = task["spec"]; appearance = spec["appearance"]
    canvas, left, top, u = board(spec, appearance)
    d = canvas.d; styles = appearance["style"]
    fills = ["#efd195", "#c0dee2", "#c6dfbf", "#e1c9e6", "#f1c4b0"]
    color = fills[appearance["palette"]]
    for loc, tile in state.items():
        r, c = rc(loc); x, y = left+c*u, top+r*u
        box = (x+8, y+8, x+u-8, y+u-8)
        if tile is None:
            d.rectangle(box, fill="#8298a3", outline="#526a76", width=3)
            canvas.entity("blank", "blank", box, None)
            canvas.probe((x+u/2, y+u/2), "#8298a3")
        else:
            if styles == "bevel":
                d.rounded_rectangle((x+10, y+12, x+u-4, y+u-3), 10, fill=tint(color, -45))
            if styles == "outline":
                d.rectangle(box, fill=color, outline="#304e5d", width=4)
            else: d.rounded_rectangle(box, 12, fill=color, outline="#526974", width=3)
            label = canvas.label((x+u/2, y+u/2), tile[1:], 58 if u < 130 else 68)
            canvas.entity(tile, "tile", box, label)
            canvas.probe((x+16, y+16), color)
    return canvas.finish()


def rush(task, state):
    spec = task["spec"]; appearance = spec["appearance"]
    canvas, left, top, u = board(spec, appearance, extra=82)
    d = canvas.d
    palette = [COLORS[c] for c in ["blue", "yellow", "green", "purple"]]
    for i, (v, anchor) in enumerate(state.items()):
        if anchor is None: continue
        r, c = rc(anchor); car = spec["cars"][v]; horizontal = car["axis"] == "horizontal"
        x, y = left+c*u+8, top+r*u+8
        w, h = (car["length"] if horizontal else 1)*u-16, (1 if horizontal else car["length"])*u-16
        color = COLORS["red"] if v == "v0" else palette[(i+appearance["palette"]) % len(palette)]
        if appearance["style"] == "outline": d.rectangle((x, y, x+w, y+h), fill=color, outline="#263f50", width=4)
        else: d.rounded_rectangle((x, y, x+w, y+h), 16, fill=color, outline="#263f50", width=4)
        if appearance["style"] == "bevel":
            if horizontal:
                for xx in [x+15, x+w-24]: d.ellipse((xx, y+h-10, xx+13, y+h+3), fill="#34454f")
                d.rounded_rectangle((x+18, y+16, x+w-18, y+h-16), 8, outline=tint(color, 35), width=7)
            else:
                for yy in [y+15, y+h-24]: d.ellipse((x+w-10, yy, x+w+3, yy+13), fill="#34454f")
                d.rounded_rectangle((x+16, y+18, x+w-16, y+h-18), 8, outline=tint(color, 35), width=7)
        for j in range(1, car["length"]):
            if horizontal: d.line((x+j*u-8, y+5, x+j*u-8, y+h-5), fill=tint(color, -35), width=2)
            else: d.line((x+5, y+j*u-8, x+w-5, y+j*u-8), fill=tint(color, -35), width=2)
        label = canvas.label((x+w/2, y+h/2), v, 38)
        canvas.entity(v, "vehicle", (x, y, x+w, y+h), label)
        canvas.probe((x+8, y+8), color)
    x = left+spec["w"]*u; y = top+(spec["target_row"]+.5)*u
    d.polygon([(x+8, y-19), (x+63, y), (x+8, y+19)], fill=COLORS["red"])
    canvas.label((x+36, y+44), "EXIT", 28)
    canvas.layout["exit_row"] = spec["target_row"]
    return canvas.finish()


def color(task, state):
    spec = task["spec"]; appearance = spec["appearance"]
    u = max(152, appearance["scale"]+32); step = 88 if appearance["scale"] < 140 else 100
    capacity = spec["capacity"]; base = 34+capacity*step
    canvas = Canvas(70+len(state)*u+24, base+78, appearance); d = canvas.d
    for j in range(capacity): canvas.label((31, base-42-j*step), f"s{j+1}", 28)
    canvas.layout["bolts"] = []
    for i, (bolt, stack) in enumerate(state.items()):
        x = 70+(i+.5)*u
        d.rounded_rectangle((x-10, 26, x+10, base), 7, fill="#becad1", outline="#7b8c98", width=2)
        d.ellipse((x-61, base-12, x+61, base+14), fill="#9dabb6", outline="#526674", width=3)
        for j in range(capacity):
            y = base-42-j*step
            d.rounded_rectangle((x-59, y-34, x+59, y+34), 8, outline="#96a2aa", width=2)
            if j < len(stack):
                nut = stack[j]; col = spec["colors"][nut]; fill = COLORS[col]
                radius = 4 if appearance["style"] == "outline" else 13
                d.rounded_rectangle((x-59, y-34, x+59, y+34), radius, fill=fill, outline=tint(fill, -40), width=3)
                if appearance["style"] == "bevel":
                    d.ellipse((x-44, y-36, x+44, y-17), fill=tint(fill, 35), outline=tint(fill, -15), width=2)
                label = canvas.label((x, y+4), nut+" "+LETTERS[col], 28)
                canvas.entity(nut, "nut", (x-59, y-34, x+59, y+34), label, column=i, color=col)
                canvas.probe((x-48, y+20), fill)
        label = canvas.label((x, base+45), bolt, 34)
        canvas.entity(bolt, "bolt", (x-62, 24, x+62, base+15), label, column=i)
        canvas.layout["bolts"].append(bolt)
    canvas.layout["capacity"] = capacity
    return canvas.finish()


def blocks(task, state):
    spec = task["spec"]; appearance = spec["appearance"]
    upper = {lower: b for b, lower in state["on"].items() if lower != "table"}
    roots = [b for b, lower in state["on"].items() if lower == "table"]
    roots.sort(key=lambda b: (spec["initial_roots"].index(b) if b in spec["initial_roots"] else 100+int(b[1:])))
    piles = []
    for root in roots:
        p = [root]
        while p[-1] in upper: p.append(upper[p[-1]])
        piles.append(p)
    u = appearance["scale"]; skew = [-.32, -.45, -.57][appearance["camera"]]
    rise = [.12, .22, .30][appearance["camera"]]; spacing = 1.5+appearance["spacing"]*.12
    height = max(map(len, piles), default=1)
    origin = (100, 44+height*u)
    def point(x, y, z): return (origin[0]+u*(x+skew*y), origin[1]+u*(rise*x+.32*y-z))
    width = int(190+max(1, len(piles))*spacing*u)
    bottom = int(origin[1]+u*(rise*max(1, len(piles))*spacing+.65)+60)
    canvas = Canvas(width+120, bottom, appearance); d = canvas.d
    extent = max(1, len(piles))*spacing-.35
    d.polygon([point(-.25, -.2, 0), point(extent, -.2, 0), point(extent, 1.15, 0), point(-.25, 1.15, 0)], fill="#b9a78b", outline="#72685a", width=3)
    d.polygon([point(-.25, 1.15, 0), point(extent, 1.15, 0), point(extent, 1.15, -.22), point(-.25, 1.15, -.22)], fill="#95816a", outline="#72685a", width=2)
    palette = list(COLORS.values())
    def cube(x, z, b, column=None, held=False):
        fill = palette[(int(b[1:])+appearance["palette"]) % len(palette)]
        vertices = {(dx, dy, dz): point(x+dx, dy, z+dz) for dx, dy, dz in itertools.product([0, .94], repeat=3)}
        def face(coords, col):
            d.polygon([vertices[v] for v in coords], fill=col, outline="#2c4556", width=3)
        face([(0, 0, 0), (.94, 0, 0), (.94, 0, .94), (0, 0, .94)], tint(fill, -15))
        face([(.94, 0, 0), (.94, .94, 0), (.94, .94, .94), (.94, 0, .94)], tint(fill, -32))
        face([(0, .94, 0), (.94, .94, 0), (.94, .94, .94), (0, .94, .94)], fill)
        face([(0, 0, .94), (.94, 0, .94), (.94, .94, .94), (0, .94, .94)], tint(fill, 35))
        if appearance["style"] in {"bevel", "outline"}:
            d.line([point(x+.08, .95, z+.08), point(x+.86, .95, z+.08), point(x+.86, .95, z+.85)], fill=tint(fill, 40), width=3)
        xy = point(x+.47, .96, z+.44)
        label = canvas.label(xy, b, 34 if u < 130 else 38)
        bbox = [min(p[0] for p in vertices.values()), min(p[1] for p in vertices.values()), max(p[0] for p in vertices.values()), max(p[1] for p in vertices.values())]
        canvas.entity(b, "block", bbox, label, column=column, held=held)
        canvas.probe(point(x+.08, .96, z+.40), fill)
    for i, pile in enumerate(piles):
        for z, b in enumerate(pile): cube(i*spacing, z, b, i)
    canvas.label((width/2, bottom-31), "table", 30)
    # A separate gripper area prevents the hand or held block occluding a tower.
    gx, gy = width+42, bottom-136
    d.line((gx-37, gy-65, gx-37, gy+4, gx-17, gy+19), fill="#688698", width=8)
    d.line((gx+37, gy-65, gx+37, gy+4, gx+17, gy+19), fill="#688698", width=8)
    canvas.label((gx, gy+48), "hand1", 28)
    if state["held"]:
        b = state["held"]; fill = palette[(int(b[1:])+appearance["palette"]) % len(palette)]
        d.rectangle((gx-25, gy-47, gx+25, gy-5), fill=fill, outline="#2c4556", width=2)
        label = canvas.label((gx, gy-27), b, 28)
        canvas.entity(b, "block", (gx-25, gy-47, gx+25, gy-5), label, column=None, held=True)
    return canvas.finish()


def kitchen(task, state):
    spec = task["spec"]; appearance = spec["appearance"]
    canvas, left, top, u = board(spec, appearance, unit=appearance["scale"]+100)
    d, theme = canvas.d, canvas.theme
    def origin(loc):
        r, c = rc(loc); return left+c*u, top+r*u
    for loc, box in canvas.regions.items():
        if loc not in spec["floors"]:
            x, y, a, b = box
            d.rectangle((x+2, y+2, a-2, b-2), fill=theme["wall"] if loc != spec["counter"] else "#b49b74")
    for loc, name in [(spec["chop"], "CHOP"), (spec["delivery"], "SERVE"), (spec["counter"], "pass1")]:
        x, y = origin(loc); canvas.label((x+u/2, y+25), name, 30)
    canvas.layout["counter"] = spec["counter"]
    canvas.layout["chop"] = spec["chop"]; canvas.layout["delivery"] = spec["delivery"]
    grouped = defaultdict(list)
    for f, (rel, loc) in state["food_places"].items():
        if rel == "at": grouped[loc].append(f)
        elif rel == "on" and loc == "pass1": grouped[spec["counter"]].append(f)
    def food(f, xy, where, size=34):
        x, y = xy; kind = spec["food_kinds"][f]
        fill = {"tomato": "#dc6055", "lettuce": "#77a862", "onion": "#bd83b1"}[kind]
        compact = where == "p1" or (where and where.startswith("hand"))
        radius = 16 if compact else round(u*.13)
        label_offset = 33 if compact else radius+17
        if kind == "lettuce":
            for dx, dy in [(-7, 0), (7, 0), (0, -6)]: d.ellipse((x+dx-radius, y+dy-radius, x+dx+radius, y+dy+radius), fill=fill, outline=tint(fill, -40), width=2)
        else: d.ellipse((x-radius, y-radius, x+radius, y+radius), fill=fill, outline=tint(fill, -45), width=2)
        prepared = f in state["prepared"]
        if prepared:
            d.line((x-13, y-8, x+13, y+8), fill="#fffefa", width=6)
            canvas.probe((x, y), "#fffefa")
        else: canvas.probe((x, y), fill)
        label = canvas.label((x, y+label_offset), f, size)
        extent = radius+8 if kind == "lettuce" else radius
        canvas.entity(f, "food", (x-extent, y-extent, x+extent, y+extent), label, container=where, prepared=prepared)
    for loc, items in grouped.items():
        x, y = origin(loc)
        for i, f in enumerate(items):
            food(f, (x+u*(i+1)/(len(items)+1), y+u*.36), "pass1" if loc == spec["counter"] else None)
    plate_loc = state["plate_place"][1] if state["plate_place"][0] == "at" else None
    if plate_loc:
        x, y = origin(plate_loc)
        foods = [f for f, place in state["food_places"].items() if place == ["on", "p1"]]
        plate_bottom = .80 if foods else .67
        d.ellipse((x+u*.08, y+40, x+u*.92, y+u*plate_bottom), fill="#f5faff", outline="#5e879e", width=4)
        label = canvas.label((x+u*(.18 if foods else .5), y+60), "p1", 29)
        canvas.entity("p1", "plate", (x+u*.08, y+40, x+u*.92, y+u*plate_bottom), label)
        for i, f in enumerate(foods):
            food(f, (x+u*(.47+.31*(i % 2)), y+u*(.31+.34*(i//2))), "p1", 28)
    for chef, loc in state["chefs"].items():
        x, y = origin(loc); cx, cy = x+u*.20, y+u*.85
        fill = "#417ca8" if chef == "a1" else "#ae784c"
        half = u*.16
        d.rounded_rectangle((cx-half, cy-u*.15, cx+half, cy+u*.10), 10, fill=fill, outline="#294b60", width=3)
        d.ellipse((cx-half, cy-u*.22, cx+half, cy-u*.08), fill="#f7fafd", outline="#43667c", width=2)
        label = canvas.label((cx, cy), chef, 32)
        canvas.entity(chef, "chef", (cx-half, cy-u*.22, cx+half, cy+u*.10), label)
        hand = spec["hands"][chef]; held = state["holding"][hand]
        if held:
            hx, hy = x+u*.74, y+u*.77
            d.rounded_rectangle((hx-38, hy-34, hx+38, hy+51), 8, fill="#dbe9ee", outline="#2c617c", width=3)
            if held == "p1":
                d.ellipse((hx-27, hy-17, hx+27, hy+17), fill="#f5faff", outline="#5e879e", width=3)
                label = canvas.label((hx, hy+30), "p1", 28)
                canvas.entity("p1", "plate", (hx-27, hy-17, hx+27, hy+17), label, container=hand)
            else: food(held, (hx, hy), hand, 30)
    # Contents of a held plate remain visible as labeled food badges in SERVE's
    # region only for review of the terminal state (reference plans deliver it).
    canvas.layout["served"] = state["served"]
    return canvas.finish()


RENDERERS = dict(cooperative_kitchen=kitchen, rush_hour=rush, color_sort=color,
                 blocks_3d=blocks, numeric_sliding=numeric)


def render(task, state=None):
    return RENDERERS[task["game"]](task, task["state"] if state is None else state)


def decode(task, visual):
    """Recover dynamic facts from rendered object IDs and drawing geometry.

    This path does not consult task['state'] or PDDL init. Raster checks below
    separately verify every label and color/status probe in the saved PNG.
    """
    entities, labels, layout = visual["entities"], visual["labels"], visual["layout"]
    for e in entities:
        if e["label"] is not None:
            actual = labels[e["label"]]["text"].split()[0]
            if e["kind"] == "tile": actual = "t"+actual
            assert actual == e["id"], (e["id"], actual)
    def location(e):
        a, b, c, d = e["bbox"]; grid = layout["grid"]
        r = int(((b+d)/2-grid["top"])/grid["unit"]); col = int(((a+c)/2-grid["left"])/grid["unit"])
        assert 0 <= r < grid["h"] and 0 <= col < grid["w"]
        return cell(r, col)
    g = task["game"]; spec = task["spec"]
    if g == "numeric_sliding":
        return {("empty", location(e)) if e["kind"] == "blank" else ("at", e["id"], location(e)) for e in entities}
    if g == "rush_hour":
        occupied = set(); out = set(); grid = layout["grid"]
        for e in entities:
            a, b, c, d = e["bbox"]
            r = round((b-grid["top"]-8)/grid["unit"]); col = round((a-grid["left"]-8)/grid["unit"])
            height = round((d-b+16)/grid["unit"]); width = round((c-a+16)/grid["unit"])
            assert (height == 1) != (width == 1)
            car = spec["cars"][e["id"]]
            assert max(height, width) == car["length"] and (width > 1) == (car["axis"] == "horizontal")
            out.add(("at", e["id"], cell(r, col)))
            occupied |= {cell(r+i, col+j) for i in range(height) for j in range(width)}
        if not any(e["id"] == "v0" for e in entities): out.add(("escaped", "v0"))
        out |= {("empty", cell(r, c)) for r in range(grid["h"]) for c in range(grid["w"]) if cell(r, c) not in occupied}
        return out
    if g in {"color_sort", "blocks_3d"}:
        groups = defaultdict(list); out = set()
        for e in entities:
            if e["kind"] in {"nut", "block"}: groups[e["column"]].append(e)
        if g == "color_sort":
            for i, bolt in enumerate(layout["bolts"]):
                items = sorted(groups[i], key=lambda e: -(e["bbox"][1]+e["bbox"][3]))
                out.add(("top", bolt, f"{bolt}s{len(items)}"))
                for j, e in enumerate(items, 1):
                    assert e["color"] == spec["colors"][e["id"]]
                    assert labels[e["label"]]["text"].split()[1] == LETTERS[e["color"]]
                    out |= {("in", e["id"], bolt), ("at", e["id"], f"{bolt}s{j}")}
        else:
            assert len(groups[None]) <= 1
            for column, items in groups.items():
                if column is None:
                    for e in items: out.add(("holding", "hand1", e["id"]))
                else:
                    items.sort(key=lambda e: -(e["bbox"][1]+e["bbox"][3]))
                    previous = "table"
                    for e in items: out.add(("on", e["id"], previous)); previous = e["id"]
                    out.add(("clear", previous))
            if not groups[None]: out.add(("hand_free", "hand1"))
        return out
    out = set(); held = set(); counter_used = False
    for e in entities:
        ident = e["id"]; container = e.get("container")
        if container and container.startswith("hand"):
            out.add(("holding", container, ident)); held.add(container)
        elif container:
            out.add(("on", ident, container)); counter_used |= container == "pass1"
        else: out.add(("at", ident, location(e)))
        if e.get("prepared"): out.add(("prepared", ident))
    for h in spec["hands"].values():
        if h not in held: out.add(("hand_free", h))
    if not counter_used: out.add(("counter_free", "pass1"))
    if layout["served"]: out.add(("served", "p1"))
    return out


def audit_image(task, image, visual, state=None, training=True):
    """Check every visible ID, text crop, status/color pixel and physical state."""
    width, height = image.size; labels = visual["labels"]
    for item in labels + visual["entities"]:
        a, b, c, d = item["bbox"]
        assert 0 <= a <= c <= width and 0 <= b <= d <= height, ("clipped", item)
    for i, a in enumerate(labels):
        assert a["font_size"] >= 28
        for b in labels[i+1:]:
            x, y, z, w = a["bbox"]; u, v, p, q = b["bbox"]
            assert min(z, p) <= max(x, u) or min(w, q) <= max(y, v), ("overlapping text", a["text"], b["text"])
        # A saved raster, not just metadata, must contain the intended label.
        box = a["bbox"]; x, y, z, w = box
        expected = Image.new("RGB", (z-x, w-y), a["background"])
        d = ImageDraw.Draw(expected)
        d.text((a["xy"][0]-x, a["xy"][1]-y), a["text"], font=font(a["font_size"], True), fill=a["ink"], anchor="mm")
        actual = np.asarray(image.crop(box)); target = np.asarray(expected)
        assert np.array_equal(actual, target), ("incorrect or obscured label", a["text"])
    for probe in visual["probes"]:
        expected = Image.new("RGB", (1, 1), probe["color"]).getpixel((0, 0))
        assert image.getpixel(tuple(probe["xy"])) == expected, ("obscured state/color", probe)
    shown = decode(task, visual)
    expected = projection(task, task["state"] if state is None else state)
    assert shown == expected, ("visual state mismatch", shown ^ expected)
    # IDs must remain large enough even when large scenes are displayed at 1024px.
    effective_font = min(x["font_size"] for x in labels)*min(1, 1024/max(image.size))
    if training: assert effective_font >= 15, ("too many objects for readable canvas", image.size, effective_font)
    return dict(all_labels_in_bounds=True, no_label_overlap=True, every_text_crop_verified=True,
                all_color_status_probes_verified=True, decoded_visual_state_equals_PDDL=True,
                minimum_font_pixels=min(x["font_size"] for x in labels),
                minimum_font_at_1024=round(effective_font, 2))
