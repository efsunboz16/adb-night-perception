# Faz 6 · Son ölçümler, doğrulama matrisi, demo videosu ve README görselleri (tek komut).
#
# Önkoşul: deploy/models/full_fp16.engine ve band_fp16.engine (python deploy/export.py),
#          demo videosu data/demo/ altında (ör. Pexels 15119004, 1920x1080).
# Kullanım (proje klasöründe, .venv aktifken):
#   powershell -ExecutionPolicy Bypass -File scripts/faz6_final.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/faz6_final.ps1 -Video data/demo/baska.mp4 -GifStart 5
param(
    [string]$Video = "data/demo/pexels_15119004.mp4",
    [double]$Hfov = 70,
    [double]$GifStart = 12,
    [double]$GifDuration = 10
)

$full = "deploy/models/full_fp16.engine"
$band = "deploy/models/band_fp16.engine"

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "HATA: $name (kod $LASTEXITCODE)" -ForegroundColor Red; exit $LASTEXITCODE }
}

if (-not (Test-Path $full) -or -not (Test-Path $band)) { Write-Host "Motorlar yok: python deploy/export.py" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $Video)) { Write-Host "Video yok: $Video" -ForegroundColor Red; exit 1 }
$stem = [System.IO.Path]::GetFileNameWithoutExtension($Video)

Step "1/6 Video sistem metrikleri (uzun far kullanımı, titreme)" { python eval/eval_video.py --weights $full --band-weights $band --backend trt --scenes highway residential }
Step "2/6 Kamaşma ihlali (tüm gece videoları)"                  { python eval/diagnose_glare.py --weights $full --band-weights $band --backend trt }
Step "3/6 Gereksinim doğrulama matrisi"                         { python eval/verify_requirements.py }
Step "4/6 Demo videosu"                                         { python scripts/render_demo.py --source $Video --hfov $Hfov --out "runs/demo/$stem.mp4" }
Step "5/6 README görselleri (GIF, kareler, doğrulama özeti)"    { python scripts/make_readme_media.py --demo "runs/demo/$stem.mp4" --start $GifStart --duration $GifDuration }
Step "6/6 Statik şekiller"                                      { python scripts/make_figures.py }

Write-Host "`nBitti. Gönderilecek dosyalar:" -ForegroundColor Green
$files = @("docs/verification.md", "runs/trt/full_fp16_gpu/verification.json", "runs/trt/full_fp16_gpu/eval_video.json",
           "runs/trt/full_fp16_gpu/glare_diag/rates.json", "docs/media/demo.gif", "docs/media/demo_frames.jpg",
           "docs/media/verification.png", "runs/demo/$stem.mp4")
$files | ForEach-Object { if (Test-Path $_) { "  $_ ($([math]::Round((Get-Item $_).Length / 1MB, 1)) MB)" } else { "  YOK: $_" } }
Compress-Archive -Path ($files | Where-Object { (Test-Path $_) -and ($_ -notlike "runs/demo/*") }) -DestinationPath faz6_sonuclar.zip -Force
Write-Host "faz6_sonuclar.zip hazır (demo .mp4 hariç; onu ayrıca gönder veya GitHub'a yükle)."
