#!/bin/bash
CMD="./bin/uv -t testcard:512:512:30:UYVY -M 3D -m 9000 -d gl:fs"

COMPRESS=(JPEG RTDXT uncompressed)
JPEGOPTS=(70 80 90 95 98 99 100)
DXTOPTS=(DXT1 DXT5)

export DISPLAY=:0

echo "PID $$ (report is $$.log)"

while :
do
        index=$(($RANDOM % ${#COMPRESS[*]}))
        compression=${COMPRESS[$index]}
        case $compression in
                JPEG)
                        opts=${JPEGOPTS[$(($RANDOM % ${#JPEGOPTS[*]}))]}
                        ;;
                RTDXT)
                        opts=${DXTOPTS[$(($RANDOM % ${#DXTOPTS[*]}))]}
                        ;;
        esac

	if [ $compression != "uncompressed" ]; then
		echo $CMD -c $compression:$opts 2>&1 |tee $$.log
		$CMD -c $compression:$opts &>> $$.log
	else
		echo $CMD 2>&1 |tee $$.log
		$CMD &>> $$.log
	fi
	sleep 1
done
