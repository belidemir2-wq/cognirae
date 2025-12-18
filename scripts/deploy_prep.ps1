Param()
$requiredPkgs = @("streamlit","openai","reportlab")
$errors = @()

if (-not (Test-Path "app.py")) { $errors += "Missing app.py in project root." }
if (-not (Test-Path "requirements.txt")) {
    $errors += "Missing requirements.txt (run: pip freeze > requirements.txt)"
} else {
    foreach ($pkg in $requiredPkgs) {
        if (-not (Select-String -Path "requirements.txt" -Pattern $pkg -Quiet)) {
            Write-Warning "requirements.txt: add $pkg if your app uses it."
        }
    }
}

if (Test-Path ".env") { Write-Warning ".env present. Ensure it is NOT committed and is in .gitignore." }
if (-not (Test-Path ".gitignore")) { Write-Warning "Missing .gitignore. Add one that ignores .env, .venv/, __pycache__/." }

if ($errors.Count -gt 0) {
    Write-Host "[31mNot ready:[0m"
    $errors | ForEach-Object { Write-Host " - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "`nReady to push checklist:" -ForegroundColor Green
Write-Host " - app.py present"
Write-Host " - requirements.txt present and has needed packages"
Write-Host " - .gitignore excludes .env, .venv/, __pycache__/"
Write-Host " - No secrets committed (.env stays local)"
Write-Host " - Optional local test: streamlit run app.py"
