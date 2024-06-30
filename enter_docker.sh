#脚本的当前目录
SELF_DIR="$(dirname "$(realpath "${0}")")"

docker run \
    -it \
    --gpus all \
    --user $(id -u):$(id -g) \
    --volume "${SELF_DIR}:${SELF_DIR}" \
    --workdir "${SELF_DIR}" \
    tiltrec:v1 
