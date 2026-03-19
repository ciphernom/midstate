import os
import re
import sys

# Configuration: Context-aware rules with syntax-safe fixing
RULES = {
    "InputReveal": {
        "check": r"InputReveal\s*\{",
        # Ensures a comma exists after salt before adding commitment
        "fix": lambda line: re.sub(r"(salt: [^,}]+)(\s*})", r"\1, commitment: None \2", line)
    },
    "ExecContext": {
        "check": r"ExecContext\s*\{",
        # Fixes one-liners by adding a comma before input_value
        "fix": lambda line: line.replace("}", ", input_value: 0 }") if "input_value" not in line and "}" in line and "{" in line else 
                            line.replace("}", "    input_value: 0,\n                }")
    },
    "ProverNew": {
        "check": r"ConfidentialTransferProver::new",
        # Fixes the 4-arg to 5-arg constructor mismatch
        "fix": lambda line: line.replace("]);", "], 0);") if "], 0);" not in line else line
    }
}

def parse_cargo_errors(input_text):
    error_pattern = re.compile(r"--> ([\w/.-]+\.rs):(\d+):(\d+)")
    targets = []
    for line in input_text.splitlines():
        match = error_pattern.search(line)
        if match:
            targets.append({"file": match.group(1), "line": int(match.group(2))})
    return targets

def apply_patches(targets):
    file_map = {}
    for t in targets:
        file_map.setdefault(t['file'], []).append(t['line'])

    for filepath, lines in file_map.items():
        if not os.path.exists(filepath): continue
        with open(filepath, 'r') as f:
            content = f.readlines()

        changed = False
        for line_num in sorted(set(lines), reverse=True):
            idx = line_num - 1
            found_rule = None
            target_idx = -1
            
            # Scan window
            for offset in range(-2, 3):
                check_idx = idx + offset
                if 0 <= check_idx < len(content):
                    curr_line = content[check_idx]
                    for name, rule in RULES.items():
                        if re.search(rule['check'], curr_line):
                            found_rule = rule
                            target_idx = check_idx
                            break
                if found_rule: break

            if found_rule:
                old_line = content[target_idx]
                new_line = found_rule['fix'](old_line)
                if old_line != new_line:
                    content[target_idx] = new_line
                    print(f"FIXED: {filepath}:{target_idx+1}")
                    changed = True

        if changed:
            with open(filepath, 'w') as f:
                f.writelines(content)

if __name__ == "__main__":
    print("Paste the 'cargo check' or 'cargo test' error output and press Ctrl+D:")
    cargo_output = sys.stdin.read()
    targets = parse_cargo_errors(cargo_output)
    if targets:
        apply_patches(targets)
        print("\nPatching complete. Re-run cargo check.")
