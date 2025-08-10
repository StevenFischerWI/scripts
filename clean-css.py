import re
from collections import OrderedDict

# Read your input file
with open("/Users/steven/projects/ZenBot/ZenBot.Screener.Web/Client/wwwroot/css/app.css", "r") as f:
    css = f.read()

# Regex to find all selector-rule blocks
pattern = re.compile(r'([^{]+)\{([^}]+)\}', re.DOTALL)

# OrderedDict to group by selector while keeping insertion order
grouped = OrderedDict()

matches = pattern.findall(css)
order_seen = []

for selector, body in matches:
    selector = selector.strip()
    body = body.strip()
    if selector not in grouped:
        grouped[selector] = []
        order_seen.append(selector)
    grouped[selector].append(body)

# Write output keeping only attributes whose value differs from any identical
# attribute written previously for the same selector.
with open("/Users/steven/projects/ZenBot/ZenBot.Screener.Web/Client/wwwroot/css/app.cleaned.css", "w") as f:
    for selector in order_seen:
        seen_props = {}  # prop -> last written value
        written_blocks = set()  # set of declaration tuples already output for this selector
        for body in grouped[selector]:
            # Filter declarations whose value is unchanged
            filtered_decls = []
            for decl in re.split(r';\s*', body):
                decl = decl.strip()
                if not decl or ':' not in decl:
                    continue
                prop, val = map(str.strip, decl.split(':', 1))
                if prop in seen_props and seen_props[prop] == val:
                    continue  # duplicate with same value, skip
                seen_props[prop] = val
                filtered_decls.append(f"{prop}: {val}")
            if not filtered_decls:
                continue  # this block adds nothing new
            decl_key = tuple(filtered_decls)
            if decl_key in written_blocks:
                continue  # identical declaration set already written
            written_blocks.add(decl_key)
            joined = ';\n  '.join(filtered_decls) + ';'
            f.write(f"{selector} {{\n  {joined}\n}}\n\n")

print("✅ Grouped CSS written to '/Users/steven/projects/ZenBot/ZenBot.Screener.Web/Client/wwwroot/css/app.cleaned.css'")
