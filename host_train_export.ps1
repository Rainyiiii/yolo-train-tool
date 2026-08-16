param(
    [string]$VmUser = "",
    [string]$VmHost = "",
    [string]$VmWorkDir = "~/douzi_maixcam_jobs",
    [int]$ImgSize = 448,
    [int]$Epochs = 100,
    [int]$Batch = 16,
    [double]$Lr0 = 0.01,
    [string]$CondaEnv = "yolov8",
    [string]$BaseModel = "yolov8n.pt",
    [string]$ProjectName = "douzi_yolov8n_448",
    [string]$DatasetRoot = "",
    [switch]$SkipVmConvert
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = $ScriptRoot
}
$DatasetRoot = (Resolve-Path $DatasetRoot).Path
$Root = $DatasetRoot
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Work = Join-Path $Root ".maixcam_work_$Timestamp"
$YoloData = Join-Path $Work "yolo_dataset"
$Out = Join-Path $Root "outputs_$Timestamp"

New-Item -ItemType Directory -Force $Work, $YoloData, $Out | Out-Null

$PrepPy = Join-Path $Work "prepare_voc_yolo.py"
@'
import argparse, random, shutil, xml.etree.ElementTree as ET
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--val-ratio', type=float, default=0.2)
ap.add_argument('--seed', type=int, default=42)
args = ap.parse_args()

root = Path(args.root)
out = Path(args.out)
img_dir = root / 'images'
ann_dir = root / 'annotations'

xmls = sorted(ann_dir.glob('*.xml'))
classes = []
records = []
for x in xmls:
    tree = ET.parse(x)
    r = tree.getroot()
    filename = r.findtext('filename') or (x.stem + '.jpg')
    size = r.find('size')
    w = int(float(size.findtext('width')))
    h = int(float(size.findtext('height')))
    objs = []
    for obj in r.findall('object'):
        name = (obj.findtext('name') or '').strip()
        if not name:
            continue
        if name not in classes:
            classes.append(name)
        b = obj.find('bndbox')
        xmin = max(0.0, float(b.findtext('xmin')))
        ymin = max(0.0, float(b.findtext('ymin')))
        xmax = min(float(w), float(b.findtext('xmax')))
        ymax = min(float(h), float(b.findtext('ymax')))
        if xmax <= xmin or ymax <= ymin:
            continue
        objs.append((name, xmin, ymin, xmax, ymax, w, h))
    img = img_dir / filename
    if img.exists() and objs:
        records.append((img, objs))

if not records:
    raise SystemExit('no valid image/xml records found')

random.seed(args.seed)
random.shuffle(records)
val_n = max(1, int(len(records) * args.val_ratio))
splits = {'val': records[:val_n], 'train': records[val_n:]}

for split, items in splits.items():
    (out / 'images' / split).mkdir(parents=True, exist_ok=True)
    (out / 'labels' / split).mkdir(parents=True, exist_ok=True)
    for img, objs in items:
        shutil.copy2(img, out / 'images' / split / img.name)
        lines = []
        for name, xmin, ymin, xmax, ymax, w, h in objs:
            cid = classes.index(name)
            xc = ((xmin + xmax) / 2) / w
            yc = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            lines.append(f'{cid} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}')
        (out / 'labels' / split / (img.stem + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')

names_inline = ', '.join([f'{i}: {c}' for i, c in enumerate(classes)])
(data := out / 'dataset.yaml').write_text(
    f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" + ''.join(f"  {i}: {c}\n" for i, c in enumerate(classes)),
    encoding='utf-8'
)
(out / 'classes.txt').write_text('\n'.join(classes) + '\n', encoding='utf-8')
print(f'classes={classes}')
print(f'train={len(splits["train"])} val={len(splits["val"])}')
print(data)
'@ | Set-Content -Encoding UTF8 $PrepPy

if ([string]::IsNullOrWhiteSpace($CondaEnv)) {
    $PythonCmd = @("python")
    $YoloCmd = @("yolo")
} else {
    $PythonCmd = @("conda", "run", "-n", $CondaEnv, "python")
    $YoloCmd = @("conda", "run", "-n", $CondaEnv, "yolo")
}

function Invoke-Argv($Argv, [object[]]$Rest) {
    $cmd = $Argv[0]
    $args = @()
    if ($Argv.Count -gt 1) {
        $args += $Argv[1..($Argv.Count - 1)]
    }
    $args += $Rest
    & $cmd @args
}

Invoke-Argv $PythonCmd @("-m", "pip", "install", "-U", "pip")
Invoke-Argv $PythonCmd @("-m", "pip", "install", "-U", "ultralytics", "onnx", "onnxsim", "onnxruntime", "pyyaml")
Invoke-Argv $PythonCmd @($PrepPy, "--root", $Root, "--out", $YoloData)

$DataYaml = Join-Path $YoloData "dataset.yaml"
Invoke-Argv $YoloCmd @("detect", "train", "model=$BaseModel", "data=$DataYaml", "imgsz=$ImgSize", "epochs=$Epochs", "batch=$Batch", "lr0=$Lr0", "project=$Work", "name=$ProjectName", "pretrained=True")
$BestPt = Join-Path $Work "$ProjectName\weights\best.pt"
if (!(Test-Path $BestPt)) { throw "best.pt not found: $BestPt" }
Invoke-Argv $YoloCmd @("export", "model=$BestPt", "format=onnx", "imgsz=$ImgSize", "simplify=True", "opset=17", "dynamic=False")
$BestOnnx = Join-Path $Work "$ProjectName\weights\best.onnx"
if (!(Test-Path $BestOnnx)) { throw "best.onnx not found: $BestOnnx" }

Copy-Item $BestPt (Join-Path $Out "douzi_yolov8n_448.pt") -Force
Copy-Item $BestOnnx (Join-Path $Out "douzi_yolov8n_448.onnx") -Force
Copy-Item (Join-Path $YoloData "classes.txt") (Join-Path $Out "classes.txt") -Force
New-Item -ItemType Directory -Force (Join-Path $Out "calib_images") | Out-Null
Get-ChildItem (Join-Path $YoloData "images\train") -Filter *.jpg | Select-Object -First 200 | ForEach-Object { Copy-Item $_.FullName (Join-Path $Out "calib_images" $_.Name) -Force }
$TestImage = Get-ChildItem (Join-Path $Out "calib_images") -Filter *.jpg | Select-Object -First 1
if (!$TestImage) { throw "no calibration image copied" }
Copy-Item $TestImage.FullName (Join-Path $Out "test.jpg") -Force

$VmScript = Join-Path $ScriptRoot "vm_convert_pack.sh"
if (!(Test-Path $VmScript)) { throw "vm_convert_pack.sh not found: $VmScript" }
Copy-Item $VmScript (Join-Path $Out "vm_convert_pack.sh") -Force

$Zip = Join-Path $Root "maixcam_job_$Timestamp.zip"
Compress-Archive -Path (Join-Path $Out "*") -DestinationPath $Zip -Force

Write-Host ""
Write-Host "Host output: $Out"
Write-Host "Package: $Zip"
Write-Host "Uploading to ${VmUser}@${VmHost}:$VmWorkDir ..."
ssh "${VmUser}@${VmHost}" "mkdir -p $VmWorkDir"
scp $Zip "${VmUser}@${VmHost}:$VmWorkDir/"
$ZipName = Split-Path $Zip -Leaf
$RemoteJob = "job_$Timestamp"
$RemoteOutputs = "outputs_$Timestamp"
$RemoteCmd = "cd $VmWorkDir && rm -rf $RemoteJob $RemoteOutputs ${RemoteOutputs}.tar.gz && unzip -o $ZipName -d $RemoteJob && bash $RemoteJob/vm_convert_pack.sh $RemoteJob $Timestamp"

if ($SkipVmConvert) {
    Write-Host ""
    Write-Host "VM next command:"
    Write-Host $RemoteCmd
    exit 0
}

Write-Host ""
Write-Host "Running VM conversion..."
ssh "${VmUser}@${VmHost}" $RemoteCmd

Write-Host ""
Write-Host "Downloading final outputs from VM..."
scp "${VmUser}@${VmHost}:$VmWorkDir/${RemoteOutputs}.tar.gz" $Root
$LocalTar = Join-Path $Root "${RemoteOutputs}.tar.gz"
Write-Host "Final package downloaded: $LocalTar"
Write-Host "Final local folder already contains training artifacts: $Out"
Write-Host "If you need to extract the VM package on Windows: tar -xzf $LocalTar -C $Root"
