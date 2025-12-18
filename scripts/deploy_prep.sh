#!/usr/bin/env bash
set -e
required_pkgs=("streamlit" "openai" "reportlab")
errors=()

[[ -f app.py ]] || errors+=("Missing app.py in project root.")
if [[ ! -f requirements.txt ]]; then
  errors+=("Missing requirements.txt (run: pip freeze > requirements.txt)")
else
  for pkg in "${required_pkgs[@]}"; do
    if ! grep -qi "$pkg" requirements.txt; then
      echo "Warning: requirements.txt may need $pkg"
    fi
  done
fi

if [[ -f .env ]]; then echo "Warning: .env present. Keep it out of git and list in .gitignore."; fi
if [[ ! -f .gitignore ]]; then echo "Warning: Missing .gitignore. Add .env, .venv/, __pycache__/."; fi

if [[ ${#errors[@]} -gt 0 ]]; then
  echo "Not ready:" >&2
  for e in "${errors[@]}"; do echo " - $e" >&2; done
  exit 1
fi

cat <<'EOF'
Ready to push checklist:
 - app.py present
 - requirements.txt present and has needed packages
 - .gitignore excludes .env, .venv/, __pycache__/
 - No secrets committed (.env stays local)
 - Optional local test: streamlit run app.py
EOF
