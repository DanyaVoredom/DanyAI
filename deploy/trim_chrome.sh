#!/usr/bin/env bash
ROOT=$(pgrep -f "tools/colab_cli.py" | head -1)
if [ -z "$ROOT" ]; then echo "no colab_cli"; exit 0; fi
echo "root=$ROOT"
KEEP=" $ROOT "
add=1
while [ "$add" = "1" ]; do
  add=0
  for p in $(ps -eo pid=,ppid=); do
    set -- $p
    pid=$1
    ppid=$2
    case "$KEEP" in *" $ppid "*)
      case "$KEEP" in *" $pid "*) ;; *) KEEP="$KEEP$pid "; add=1 ;; esac
      ;;
    esac
  done
done
echo "keep count $(echo $KEEP | wc -w)"
killed=0
for p in $(pgrep -x chrome; pgrep -x google-chrome; pgrep -x chrome_crashpad); do
  case "$KEEP" in *" $p "*)
    ;;
  *)
    kill -9 "$p" 2>/dev/null && killed=$((killed + 1))
    ;;
  esac
done
echo "killed $killed"
sleep 2
free -m | sed -n 2p
