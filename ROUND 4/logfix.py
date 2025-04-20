import re
import json

allowed_products = {
    'KELP', 'RAINFOREST_RESIN', 'SQUID_INK', 'JAMS', 'CROISSANTS', 'DJEMBES',
    'PICNIC_BASKET1', 'PICNIC_BASKET2',
    'VOLCANIC_ROCK', 'VOLCANIC_ROCK_VOUCHER_10000',
    'VOLCANIC_ROCK_VOUCHER_10250', 'VOLCANIC_ROCK_VOUCHER_10500',
    'VOLCANIC_ROCK_VOUCHER_9500', 'VOLCANIC_ROCK_VOUCHER_9750'
}

input_file = "log.log"
output_file = "clean_log.log"

def fix_json_line(s: str) -> str:
    # Remove trailing commas before closing brackets/braces
    s = re.sub(r',\s*(\]|\})', r'\1', s)

    # Fix bad escape sequences inside strings
    s = re.sub(r'\\(?!["\\/bfnrtu])', '', s)

    # Remove non-printables
    s = ''.join(c for c in s if c.isprintable())

    return s

def extract_largest_balanced_curly_block(s: str) -> str:
    """
    Extract the largest balanced JSON object enclosed in curly braces.
    """
    stack = []
    start = -1
    max_block = ""

    for i, c in enumerate(s):
        if c == "{":
            if not stack:
                start = i
            stack.append(i)
        elif c == "}":
            if stack:
                stack.pop()
                if not stack and start != -1:
                    block = s[start:i+1]
                    if len(block) > len(max_block):
                        max_block = block
    return max_block

def try_parse_json(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def fix_and_extract_json(line: str):
    # Step 1: Cleanup malformed JSON-like strings
    fixed_line = fix_json_line(line)
    print(fixed_line)
    # Step 2: Extract balanced JSON object
    block = extract_largest_balanced_curly_block(fixed_line)
    if not block:
        return None

    # Step 3: Try parse
    parsed = try_parse_json(block)
    if parsed is None:
        return None

    # Step 4: Check product presence
    json_str = json.dumps(parsed)
    if any(product in json_str for product in allowed_products):
        return json_str

    return None

# Main loop
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        cleaned = fix_and_extract_json(line)
        if cleaned:
            outfile.write(cleaned + "\n")
