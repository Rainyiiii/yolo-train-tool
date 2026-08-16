#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="${1:-}"
TS="${2:-$(date +%Y%m%d_%H%M%S)}"
IMG_WIDTH="${IMG_WIDTH:-${IMG_SIZE:-448}}"
IMG_HEIGHT="${IMG_HEIGHT:-${IMG_SIZE:-448}}"
MODEL_NAME="${MODEL_NAME:-douzi_yolov8n_448}"
OPERATOR_MODE="${OPERATOR_MODE:-recommended}"
CONTAINER_NAME="${CONTAINER_NAME:-tpu-env-douzi}"
IMAGE_NAME="${IMAGE_NAME:-sophgo/tpuc_dev:latest}"

if [ -z "$JOB_DIR" ]; then
  echo "usage: bash vm_convert_pack.sh <job_dir> [timestamp]"
  exit 1
fi

JOB_DIR="$(readlink -f "$JOB_DIR")"
OUT_DIR="$(pwd)/outputs_${TS}"
mkdir -p "$OUT_DIR"
cp "$JOB_DIR/${MODEL_NAME}.onnx" "$OUT_DIR/"
cp "$JOB_DIR/classes.txt" "$OUT_DIR/"
cp "$JOB_DIR/test.jpg" "$OUT_DIR/"
rm -rf "$OUT_DIR/calib_images"
cp -r "$JOB_DIR/calib_images" "$OUT_DIR/calib_images"

cat > "$OUT_DIR/convert_inside_container.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
cd /workspace

IMG_WIDTH="${IMG_WIDTH:-${IMG_SIZE:-448}}"
IMG_HEIGHT="${IMG_HEIGHT:-${IMG_SIZE:-448}}"
MODEL_NAME="${MODEL_NAME:-douzi_yolov8n_448}"
OPERATOR_MODE="${OPERATOR_MODE:-recommended}"
ONNX="${MODEL_NAME}.onnx"
CVIMODEL="${MODEL_NAME}_int8.cvimodel"
echo "Operator mode: ${OPERATOR_MODE}"

if ! command -v model_transform.py >/dev/null 2>&1; then
  pip install -U tpu_mlir
fi

python3 - <<'PY'
import onnx
m = onnx.load('douzi_yolov8n_448.onnx')
print('ONNX graph outputs:')
for o in m.graph.output:
    print(o.name)
PY

model_transform.py \
  --model_name ${MODEL_NAME} \
  --model_def ./${ONNX} \
  --input_shapes [[1,3,${IMG_HEIGHT},${IMG_WIDTH}]] \
  --mean "0,0,0" \
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098" \
  --keep_aspect_ratio \
  --pixel_format rgb \
  --channel_format nchw \
  --output_names "/model.22/Concat_1_output_0,/model.22/Concat_2_output_0,/model.22/Concat_3_output_0" \
  --test_input ./test.jpg \
  --test_result ${MODEL_NAME}_top_outputs.npz \
  --tolerance 0.99,0.99 \
  --mlir ${MODEL_NAME}.mlir

CALIB_NUM=$(find ./calib_images -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n 200 | wc -l)
if [ "$CALIB_NUM" -lt 1 ]; then
  echo "no calibration images found"
  exit 1
fi

run_calibration.py ${MODEL_NAME}.mlir \
  --dataset ./calib_images \
  --input_num ${CALIB_NUM} \
  -o ${MODEL_NAME}_cali_table

model_deploy.py \
  --mlir ${MODEL_NAME}.mlir \
  --quantize INT8 \
  --quant_input \
  --calibration_table ${MODEL_NAME}_cali_table \
  --processor cv181x \
  --test_input ${MODEL_NAME}_in_f32.npz \
  --test_reference ${MODEL_NAME}_top_outputs.npz \
  --tolerance 0.9,0.6 \
  --model ${CVIMODEL}
EOS
chmod +x "$OUT_DIR/convert_inside_container.sh"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run --privileged --name "$CONTAINER_NAME" \
  -v "$OUT_DIR:/workspace" \
  -e IMG_WIDTH="$IMG_WIDTH" \
  -e IMG_HEIGHT="$IMG_HEIGHT" \
  -e MODEL_NAME="$MODEL_NAME" \
  "$IMAGE_NAME" \
  bash /workspace/convert_inside_container.sh

LABELS=$(paste -sd ', ' "$OUT_DIR/classes.txt" | sed 's/, /, /g')
cat > "$OUT_DIR/${MODEL_NAME}.mud" <<EOF2
[basic]
type = cvimodel
model = ${MODEL_NAME}_int8.cvimodel

[extra]
model_type = yolov8
input_type = rgb
mean = 0, 0, 0
scale = 0.00392156862745098, 0.00392156862745098, 0.00392156862745098
labels = ${LABELS}
EOF2

FINAL_DIR="$(pwd)/outputs_${TS}"
mkdir -p "$FINAL_DIR"
cp "$OUT_DIR/${MODEL_NAME}_int8.cvimodel" "$FINAL_DIR/"
cp "$OUT_DIR/${MODEL_NAME}.mud" "$FINAL_DIR/"
cp "$OUT_DIR/classes.txt" "$FINAL_DIR/"
cp "$OUT_DIR/${MODEL_NAME}.onnx" "$FINAL_DIR/"

tar -czf "${FINAL_DIR}.tar.gz" -C "$(dirname "$FINAL_DIR")" "$(basename "$FINAL_DIR")"
echo "Done: $FINAL_DIR"
echo "Archive: ${FINAL_DIR}.tar.gz"
